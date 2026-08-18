#!/usr/bin/env python3
"""
core/prpd_processor.py
-----------------------
Traitement signal PRPD — adapté de PRPD_sync_RX.py.

Ajoute la conversion dBFS → dBm via la calibration.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

# ──────────────────────────────────────────────────────────────
# Paramètres PRPD
# ──────────────────────────────────────────────────────────────

PRPD_NOISE_STD_MULT: float = 2.0   # seuil = médiane + N*std
PRPD_MIN_DISTANCE_US: float = 200.0  # µs entre deux pulses
PRPD_CUTOFF_HZ: float = 5e6         # filtre passe-bas IF
PRPD_FILTER_ORDER: int = 4
EPSILON: float = 1e-12


# ──────────────────────────────────────────────────────────────
# Synchronisation PPS 50 Hz
# ──────────────────────────────────────────────────────────────

def sync_on_external_50hz_pps(usrp) -> None:
    """
    Synchronise l'horloge USRP sur le front montant PPS externe (50 Hz injecté).
    Doit être appelé avant chaque acquisition PRPD.
    """
    usrp.set_clock_source("internal")
    usrp.set_time_source("external")

    t_last = usrp.get_time_last_pps().get_real_secs()
    while usrp.get_time_last_pps().get_real_secs() == t_last:
        time.sleep(0.001)

    usrp.set_time_next_pps(
        __import__("uhd").types.TimeSpec(0.0)
    )
    time.sleep(0.05)


# ──────────────────────────────────────────────────────────────
# Acquisition synchronisée
# ──────────────────────────────────────────────────────────────

def rx_prpd_sync(
    usrp,
    freq_hz: float,
    rate_hz: float,
    duration_s: float,
    gain_db: float,
    channel: int = 0,
    antenna: str = "RX2",
    progress_cb=None,
) -> tuple[np.ndarray, float | None]:
    """
    Acquisition PRPD synchronisée sur PPS 50 Hz.

    Retourne : (samples, t_start_s)
    progress_cb(pct: float) pour mise à jour de la barre de progression.
    """
    import uhd

    num_samps = int(duration_s * rate_hz)

    usrp.set_rx_rate(rate_hz, channel)
    usrp.set_rx_freq(uhd.types.TuneRequest(freq_hz), channel)
    usrp.set_rx_gain(gain_db, channel)
    usrp.set_rx_antenna(antenna, channel)
    usrp.set_rx_dc_offset(True, channel)
    usrp.set_rx_iq_balance(True, channel)

    time.sleep(0.5)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [channel]
    rx_streamer = usrp.get_rx_stream(stream_args)
    rx_md = uhd.types.RXMetadata()

    received_buf = np.zeros(num_samps, dtype=np.complex64)

    current_t = usrp.get_time_now().get_real_secs()
    future_t  = current_t + 0.1
    start_t   = np.ceil(future_t / 0.02) * 0.02

    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    cmd.num_samps   = num_samps
    cmd.stream_now  = False
    cmd.time_spec   = uhd.types.TimeSpec(start_t)
    rx_streamer.issue_stream_cmd(cmd)

    buff = np.zeros((1, 4096), dtype=np.complex64)
    total = 0
    t_start = None

    while total < num_samps:
        n = rx_streamer.recv(buff, rx_md, timeout=5.0)

        if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
            if rx_md.error_code == uhd.types.RXMetadataErrorCode.late_command:
                break
            continue

        if n > 0:
            if t_start is None:
                t_start = rx_md.time_spec.get_real_secs()
            end = min(total + n, num_samps)
            received_buf[total:end] = buff[0, : end - total]
            total = end

            if progress_cb is not None:
                progress_cb(int(total / num_samps * 100))

    return received_buf[:total], t_start


# ──────────────────────────────────────────────────────────────
# Traitement signal PRPD
# ──────────────────────────────────────────────────────────────

def process_pd_signal(
    samples: np.ndarray,
    rate_hz: float,
    t_start: float | None,
    f_offset_hz: float = 0.0,
    gain_calibration: dict | None = None,
    freq_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Détecte les pulses PD dans les samples et calcule leurs phases PRPD.

    Paramètres
    ----------
    samples         : Échantillons IQ (complex64)
    rate_hz         : Taux d'échantillonnage
    t_start         : Temps absolu du premier échantillon (secondes)
    f_offset_hz     : Décalage fréquentiel pour recentrer le signal
    gain_calibration: Dict calibration (si None → sortie en dBFS seulement)
    freq_hz         : Fréquence centrale réelle (pour calibration)

    Retourne
    --------
    phases_deg      : Phase 50 Hz de chaque pulse (0→360°)
    amplitudes      : Amplitude en dBm si calibration, sinon dBFS
    info            : Dictionnaire de diagnostic
    """
    samples = np.asarray(samples, dtype=np.complex64).ravel()
    N = len(samples)

    if N == 0 or t_start is None:
        return np.array([]), np.array([]), {"n_pulses": 0, "unit": "dBFS"}

    # Soustraction DC
    samples = samples - np.mean(samples)

    t = np.arange(N) / rate_hz

    # Translation fréquentielle vers DC (si f_offset != 0)
    if abs(f_offset_hz) > 1.0:
        samples = samples * np.exp(-1j * 2 * np.pi * f_offset_hz * t)

    # Filtre passe-bas
    sos = butter(
        PRPD_FILTER_ORDER,
        PRPD_CUTOFF_HZ / (rate_hz / 2),
        btype="low",
        output="sos",
    )
    filtered = sosfiltfilt(sos, samples)

    # Enveloppe
    envelope = np.abs(filtered)

    # Seuil dynamique
    noise_level = np.median(envelope)
    noise_std   = np.std(envelope)
    threshold   = noise_level + PRPD_NOISE_STD_MULT * noise_std

    min_dist = int(PRPD_MIN_DISTANCE_US * 1e-6 * rate_hz)

    peaks, _ = find_peaks(
        envelope,
        height=threshold,
        distance=max(1, min_dist),
    )

    if len(peaks) == 0:
        return (
            np.array([]),
            np.array([]),
            {"n_pulses": 0, "unit": "dBFS", "threshold": float(threshold)},
        )

    # Temps absolu des pulses
    t_peaks = t_start + peaks / rate_hz

    # Phase 50 Hz : 0.02 s → 360°
    phases_deg = ((t_peaks % 0.02) / 0.02) * 360.0

    # Amplitude en dBFS
    amps_raw    = envelope[peaks]
    amps_dbfs   = 20.0 * np.log10(np.clip(amps_raw, EPSILON, None))

    unit = "dBFS"
    amps_out = amps_dbfs

    # Conversion dBm si calibration disponible
    if gain_calibration is not None and freq_hz is not None:
        try:
            from core.calibration import interpolate_reference_dbm
            ref_dbm  = interpolate_reference_dbm(gain_calibration, freq_hz)
            amps_out = amps_dbfs + ref_dbm
            unit     = "dBm"
        except Exception:
            pass  # Garde dBFS en cas d'erreur de calibration

    info = {
        "n_pulses":  len(peaks),
        "unit":      unit,
        "threshold": float(threshold),
        "noise_med": float(noise_level),
    }

    return phases_deg, amps_out, info


