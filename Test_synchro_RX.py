#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import uhd
from scipy.signal import butter, sosfiltfilt, find_peaks, hilbert
import time


# ============================================================
# CONFIGURATION
# ============================================================

FREQ = 100e6
RATE = 10e6
DURATION = 0.5

GAIN_PD = 30
GAIN_SYNC = 30

CHANNEL_PD = 0
CHANNEL_SYNC = 1

PD_IF_FREQ = 2e6
SYNC_FREQ = 50.0


# ============================================================
# ACQUISITION RX0 + RX1 SYNCHRONISÉE
# ============================================================

def acquire_two_channels():
    print("Initialisation USRP...")

    usrp = uhd.usrp.MultiUSRP()

    channels = [CHANNEL_PD, CHANNEL_SYNC]

    usrp.set_rx_rate(RATE, CHANNEL_PD)
    usrp.set_rx_rate(RATE, CHANNEL_SYNC)

    usrp.set_rx_freq(uhd.types.TuneRequest(FREQ), CHANNEL_PD)
    usrp.set_rx_freq(uhd.types.TuneRequest(FREQ), CHANNEL_SYNC)

    usrp.set_rx_gain(GAIN_PD, CHANNEL_PD)
    usrp.set_rx_gain(GAIN_SYNC, CHANNEL_SYNC)

    usrp.set_rx_antenna("RX2", CHANNEL_PD)
    usrp.set_rx_antenna("RX2", CHANNEL_SYNC)

    time.sleep(0.2)

    num_samps = int(DURATION * RATE)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = channels

    rx_streamer = usrp.get_rx_stream(stream_args)
    metadata = uhd.types.RXMetadata()

    buffer_size = 4096
    buffer = np.zeros((len(channels), buffer_size), dtype=np.complex64)
    samples = np.zeros((len(channels), num_samps), dtype=np.complex64)

    # IMPORTANT :
    # Avec plusieurs channels, il ne faut PAS utiliser stream_now=True.
    # Il faut programmer un start_time pour aligner RX0 et RX1.
    usrp.set_time_now(uhd.types.TimeSpec(0.0))
    time.sleep(0.05)

    start_time = usrp.get_time_now().get_real_secs() + 0.2

    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    cmd.num_samps = num_samps
    cmd.stream_now = False
    cmd.time_spec = uhd.types.TimeSpec(start_time)

    rx_streamer.issue_stream_cmd(cmd)

    print("Acquisition en cours...")

    total = 0

    while total < num_samps:
        n = rx_streamer.recv(buffer, metadata, timeout=3.0)

        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print("Erreur RX :", metadata.strerror())
            continue

        end = min(total + n, num_samps)
        samples[:, total:end] = buffer[:, :end - total]
        total = end

    print("Acquisition terminée.")

    pd_signal = samples[0]
    sync_signal = samples[1]

    return pd_signal, sync_signal


# ============================================================
# EXTRACTION PHASE 50 Hz
# ============================================================

def extract_sync_phase(sync_signal, rate):
    sync_real = np.real(sync_signal)

    low = 40 / (rate / 2)
    high = 60 / (rate / 2)

    sos = butter(4, [low, high], btype="bandpass", output="sos")
    sync_filtered = sosfiltfilt(sos, sync_real)

    analytic = hilbert(sync_filtered)
    phase_rad = np.angle(analytic)

    phase_deg = (np.degrees(phase_rad) + 360) % 360

    return phase_deg, sync_filtered


# ============================================================
# DÉTECTION DES IMPULSIONS PD
# ============================================================

def detect_pd_events(pd_signal, rate, f_offset=PD_IF_FREQ):
    N = len(pd_signal)
    t = np.arange(N) / rate

    # Translation vers baseband
    pd_bb = pd_signal * np.exp(-1j * 2 * np.pi * f_offset * t)

    # Filtre passe-bas signal PD
    cutoff_pd = 1e6
    sos_pd = butter(4, cutoff_pd / (rate / 2), btype="low", output="sos")
    pd_filtered = sosfiltfilt(sos_pd, pd_bb)

    # Enveloppe
    envelope_raw = np.abs(pd_filtered)

    # Lissage enveloppe
    cutoff_env = 10000
    sos_env = butter(4, cutoff_env / (rate / 2), btype="low", output="sos")
    envelope = sosfiltfilt(sos_env, envelope_raw)

    # Seuil automatique
    noise_level = np.median(envelope)
    noise_std = np.std(envelope)
    threshold = noise_level + 4 * noise_std

    min_distance = int(200e-6 * rate)

    peaks, _ = find_peaks(
        envelope,
        height=threshold,
        distance=min_distance
    )

    amps = envelope[peaks]

    # dB relatif, pas dBm
    amps_db = 20 * np.log10(np.clip(amps, 1e-12, None))

    return peaks, amps_db, envelope, threshold


# ============================================================
# AFFICHAGE TEMPOREL
# ============================================================

def plot_time_debug(pd_signal, sync_filtered, envelope, threshold, rate):
    t = np.arange(len(pd_signal)) / rate

    n_show = int(0.05 * rate)

    plt.figure(figsize=(14, 6))

    plt.plot(t[:n_show], np.real(pd_signal[:n_show]), label="Signal PD réel")
    plt.plot(t[:n_show], sync_filtered[:n_show], label="Synchro 50 Hz filtrée")
    plt.plot(t[:n_show], envelope[:n_show], label="Enveloppe PD")
    plt.axhline(threshold, linestyle="--", label="Seuil détection")

    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.title("Debug temporel : PD + synchro")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# AFFICHAGE PRPD
# ============================================================

def plot_prpd(phases_pd, amps_db):
    plt.figure(figsize=(10, 6))

    plt.scatter(phases_pd, amps_db, s=15, alpha=0.7)

    plt.xlabel("Phase 50 Hz (°)")
    plt.ylabel("Amplitude PD relative (dB)")
    plt.title("Carte PRPD synchronisée")
    plt.xlim(0, 360)
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))

    plt.hist2d(phases_pd, amps_db, bins=[128, 50], cmap="jet")
    plt.colorbar(label="Densité")

    plt.xlabel("Phase 50 Hz (°)")
    plt.ylabel("Amplitude PD relative (dB)")
    plt.title("Heatmap PRPD")
    plt.xlim(0, 360)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    pd_signal, sync_signal = acquire_two_channels()

    sync_phase_deg, sync_filtered = extract_sync_phase(sync_signal, RATE)

    peaks, amps_db, envelope, threshold = detect_pd_events(
        pd_signal,
        RATE,
        f_offset=PD_IF_FREQ
    )

    print("Nombre de pics PD détectés :", len(peaks))

    if len(peaks) == 0:
        print("Aucune décharge détectée.")
        print("Vérifie le gain, le seuil, la fréquence IF PD ou le signal d'entrée.")
        return

    phases_pd = sync_phase_deg[peaks]

    plot_time_debug(
        pd_signal,
        sync_filtered,
        envelope,
        threshold,
        RATE
    )

    plot_prpd(phases_pd, amps_db)


if __name__ == "__main__":
    main()