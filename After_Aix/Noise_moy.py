import numpy as np
import matplotlib.pyplot as plt
import uhd
from scipy.signal import butter, find_peaks, sosfiltfilt
import time
import os

# =========================
# CONFIGURATION
# =========================

NFFT = 4096
FREQ = 1900e6
RATE = 12e6
DURATION = 10
GAIN = 35
CHANNEL = 0
ANTENNA = "RX2"

R = 50

# fréquence du signal utile par rapport au centre SDR
# exemple : si pic utile à 121 MHz et FREQ=116 MHz => F_OFFSET=5e6
F_OFFSET = 5e6

N_ACQ = 1

OUTPUT_DIR = "./Main_figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# SPECTRE EN dBm/bin APPROX
# =========================

def compute_time_less_noise(samples, n_pt=4096, remove_dc=True):
    samples = np.asarray(samples, dtype=np.complex64).ravel()

    if len(samples) < n_pt:
        return np.array([]), np.array([])

    # if remove_dc:
    #     samples = samples - np.mean(samples)

    n_blocks = len(samples) // n_pt
    print("nbr des block = ", n_blocks)
    samples = samples[:n_blocks * n_pt]
    blocks = samples.reshape(n_blocks, n_pt)


    p_acc = np.zeros(n_pt, dtype = np.complex128)

    for blk in blocks:

        p_acc += blk

    p_mean = p_acc / n_blocks
    print("dtype = ", type(p_mean))



    return p_mean


def plot_spectrum(freqs, psd_dbm, name="Freq_domain_plot"):
    if len(freqs) == 0:
        print("Spectre vide.")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(freqs / 1e6, psd_dbm, color="green")

    plt.title("Spectre FFT - Puissance approximative en dBm/bin")
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Puissance approximative (dBm/bin)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", bbox_inches="tight")
    plt.show()


# =========================
# TRAITEMENT PRPD EN dBm APPROX
# =========================

def process_pd_signal_dbm(samples, rate, t_start=0.0, f_offset=5e6, R=50):
    samples = np.asarray(samples, dtype=np.complex64).ravel()
    N = len(samples)

    if N == 0 or t_start is None:
        return np.array([]), np.array([])

    samples = samples - np.mean(samples)

    t = np.arange(N) / rate

    samples_dc = samples * np.exp(-1j * 2 * np.pi * f_offset * t)

    cutoff_if = 1e6
    sos_if = butter(4, cutoff_if / (rate / 2), btype="low", output="sos")
    samples_if = sosfiltfilt(sos_if, samples_dc)

    envelope_raw = np.abs(samples_if)

    cutoff_env = 10_000
    sos_env = butter(4, cutoff_env / (rate / 2), btype="low", output="sos")
    envelope = sosfiltfilt(sos_env, envelope_raw)

    noise_level = np.median(envelope)
    noise_std = np.std(envelope)

    threshold = (noise_level + 4.0 * noise_std)/3

    min_distance = int(200e-6 * rate)

    peaks, props = find_peaks(
        envelope,
        height=threshold,
        distance=min_distance
    )

    if len(peaks) == 0:
        return np.array([]), np.array([])

    t_peaks = t_start + peaks / rate

    cycle_time = t_peaks % 0.02
    phases = (cycle_time / 0.02) * 360.0

    amps = envelope[peaks]

    vrms = amps / np.sqrt(2)
    p_w = (vrms ** 2) / R

    amps_dbm = 10 * np.log10(np.clip(p_w / 1e-3, 1e-20, None))

    return phases, amps_dbm


# =========================
# AFFICHAGE PRPD
# =========================

def plot_prpd(acquisitions, name="PRPD_dBm"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["blue", "red", "green", "orange", "purple"]

    all_phases = []
    all_amps = []

    for i, (phases, amps) in enumerate(acquisitions):
        if len(phases) == 0:
            continue

        ax1.scatter(
            phases,
            amps,
            s=15,
            c=colors[i % len(colors)],
            alpha=0.6,
            label=f"Acquisition {i + 1}"
        )

        all_phases.extend(phases)
        all_amps.extend(amps)

    if len(all_amps) > 0:
        phases_ref = np.linspace(0, 360, 500)

        y_min = np.min(all_amps)
        y_max = np.max(all_amps)
        y_span = y_max - y_min if y_max > y_min else 20

        ref_50hz = np.sin(np.radians(phases_ref)) * (0.35 * y_span)
        ref_50hz += y_min + 0.5 * y_span

        ax1.plot(
            phases_ref,
            ref_50hz,
            color="black",
            linewidth=2,
            label="Référence 50 Hz"
        )

    ax1.set_title(f"Carte PRPD - {name}")
    ax1.set_xlabel("Phase (degrés)")
    ax1.set_ylabel("Puissance impulsion approximative (dBm)")
    ax1.set_xlim(0, 360)
    ax1.grid(True)
    ax1.legend(loc="best")

    if len(all_phases) > 0:
        h = ax2.hist2d(
            all_phases,
            all_amps,
            bins=[128, 50],
            cmap="jet"
        )
        fig.colorbar(h[3], ax=ax2, label="Densité")

    ax2.set_title(f"Heatmap PRPD - {name}")
    ax2.set_xlabel("Phase (degrés)")
    ax2.set_ylabel("Puissance impulsion approximative (dBm)")
    ax2.set_xlim(0, 360)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", bbox_inches="tight")
    plt.show()


# =========================
# AFFICHAGE TEMPOREL
# =========================

def plot_time_domain(samples, rate, name="time"):
    if len(samples) == 0:
        return

    t = np.arange(len(samples)) / rate

    alpha = max(1, int(len(t) / 25))

    ref_50hz = 0.0023 * np.sin(2 * np.pi * 50 * t)

    plt.figure(figsize=(12, 5))
    plt.plot(t[:alpha], np.real(samples[:alpha]), label="Partie réelle", color="blue")
    plt.plot(t[:alpha], ref_50hz[:alpha], label="Référence 50 Hz", color="red")

    plt.title("Signal temporel")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude numérique SDR")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", bbox_inches="tight")
    plt.show()


# =========================
# ACQUISITION RX SYNCHRONISÉE
# =========================

def rx_only_sync(usrp, freq, rate, duration, gain=0, antenna="RX2"):
    num_samps = int(duration * rate)

    usrp.set_rx_rate(rate, CHANNEL)
    usrp.set_rx_freq(uhd.types.TuneRequest(freq), CHANNEL)
    usrp.set_rx_gain(gain, CHANNEL)
    usrp.set_rx_antenna(antenna, CHANNEL)

    time.sleep(0.5)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [CHANNEL]

    rx_streamer = usrp.get_rx_stream(stream_args)
    rx_md = uhd.types.RXMetadata()

    received = np.zeros(num_samps, dtype=np.complex64)

    current_time = usrp.get_time_now().get_real_secs()
    future_time = current_time + 2.0

    start_time = np.ceil(future_time / 0.02) * 0.02

    print(f"Temps SDR actuel = {current_time:.6f} s")
    print(f"RX planifié à t = {start_time:.6f} s")

    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samps
    stream_cmd.stream_now = False
    stream_cmd.time_spec = uhd.types.TimeSpec(start_time)

    rx_streamer.issue_stream_cmd(stream_cmd)

    buff = np.zeros((1, 4096), dtype=np.complex64)

    total = 0
    t_start = None

    while total < num_samps:
        n = rx_streamer.recv(buff, rx_md, timeout=5.0)

        if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
            print("RX error:", rx_md.strerror())

            if rx_md.error_code == uhd.types.RXMetadataErrorCode.late_command:
                break

            continue

        if n > 0:
            if t_start is None:
                t_start = rx_md.time_spec.get_real_secs()

            end = min(total + n, num_samps)
            received[total:end] = buff[0, :end - total]
            total = end

    print(f"RX reçu : {total}/{num_samps} samples")

    if t_start is not None:
        print(f"Début réel RX : {t_start:.6f} s")
    else:
        print("Aucun sample reçu.")

    return received[:total], t_start


# =========================
# MAIN
# =========================

def main():
    print("Initialisation USRP...")

    usrp = uhd.usrp.MultiUSRP()

    print("Synchronisation PPS externe...")

    usrp.set_time_source("external")

    time_last = usrp.get_time_last_pps().get_real_secs()

    while usrp.get_time_last_pps().get_real_secs() == time_last:
        time.sleep(0.1)

    usrp.set_time_next_pps(uhd.types.TimeSpec(0.0))
    time.sleep(1.2)

    print("Synchronisé.")

    acquisitions = []

    t0 = time.time()

    for i in range(N_ACQ):
        print(f"\n--- Acquisition {i + 1}/{N_ACQ} ---")

        rx_signal, t_start = rx_only_sync(
            usrp=usrp,
            freq=FREQ,
            rate=RATE,
            duration=DURATION,
            gain=GAIN,
            antenna=ANTENNA
        )

        np.save("/home/yousef/Documents/testing_scripts/GEVernova/After_Aix/CAL2B_files/rx_signal_5V_1900Mhz_Gain35.npy", rx_signal)

        if len(rx_signal) == 0:
            print("Acquisition vide.")
            continue

        plot_time_domain(
            rx_signal,
            RATE,
            name=f"Time_domain_rx_{i + 1}"
        )
        samples_less_noise = compute_time_less_noise(rx_signal, n_pt=4096*32, remove_dc=True)
        print("Nbr de point = ", len(samples_less_noise))
        plot_time_domain(
            samples_less_noise,
            RATE,
            name=f"Time_domain_rx_{i + 1}"
        )
        # freqs, psd_dbm = compute_spectrum_dbm_per_bin(
        #     rx_signal,
        #     RATE,
        #     FREQ,
        #     nfft=NFFT,
        #     R=R,
        #     remove_dc=True
        # )

        # plot_spectrum(
        #     freqs,
        #     psd_dbm,
        #     name=f"Spectrum_dBm_bin_rx_{i + 1}"
        # )

        # phases, amps_dbm = process_pd_signal_dbm(
        #     rx_signal,
        #     RATE,
        #     t_start=t_start,
        #     f_offset=F_OFFSET,
        #     R=R
        # )

        # print(f"Nombre de pulses détectés : {len(phases)}")

        # acquisitions.append((phases, amps_dbm))

    # print(f"\nTemps total = {time.time() - t0:.2f} s")

    # plot_prpd(
    #     acquisitions,
    #     name="PRPD_dBm_approx"
    # )


if __name__ == "__main__":
    main()