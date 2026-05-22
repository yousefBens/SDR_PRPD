import numpy as np
import matplotlib.pyplot as plt
import uhd
from scipy.signal import butter, find_peaks, sosfiltfilt
import threading
import time
import os

NFFT = 4096
FREQ = 1e9 - 2e6
RATE = 30e6
DURATION = 0.5

os.makedirs("./Main_figs", exist_ok=True)


def gaussian_phase(phase_deg, center, sigma):
    return np.exp(-((phase_deg - center) ** 2) / (2 * sigma ** 2))


def Simulate_PD_Signal(num_samps, rate, f_offset=10e6):
    t = np.arange(num_samps) / rate
    pd_signal = np.zeros(num_samps, dtype=np.complex64)

    f_ref = 50.0
    phase_ref = (360.0 * f_ref * t) % 360.0

    prob_pd = (
        0.8 * gaussian_phase(phase_ref, 60, 20)
        + 1.0 * gaussian_phase(phase_ref, 200, 8)
    )
    prob_pd = prob_pd / np.max(prob_pd)

    rng = np.random.default_rng(42)
    p_global = 0.01 * prob_pd
    candidats = np.where(rng.random(num_samps) < p_global)[0]

    min_gap_samples = int(200e-6 * rate)
    event_indices = []
    last_idx = -min_gap_samples

    for c in candidats:
        if c - last_idx >= min_gap_samples:
            event_indices.append(c)
            last_idx = c

    pulse_duration = 150e-6
    pulse_len = int(pulse_duration * rate)
    tau = 30e-6
    tp = np.arange(pulse_len) / rate

    for idx in event_indices:
        A = rng.uniform(0.005, 0.04)
        pulse = A * np.exp(-tp / tau) * np.exp(1j * 2 * np.pi * f_offset * tp)

        end_idx = min(idx + pulse_len, num_samps)
        pd_signal[idx:end_idx] += pulse[:end_idx - idx]

    noise = (
        np.random.normal(0, 0.002, num_samps)
        + 1j * np.random.normal(0, 0.002, num_samps)
    )

    return pd_signal  # + noise


def Process_PD_Signal(samples, rate, t_start=0.0, f_offset=10e6, R=50):
    N = len(samples)
    t = np.arange(N) / rate

    samples_dc = samples * np.exp(-1j * 2 * np.pi * f_offset * t)

    cutoff_if = 1e6
    sos_if = butter(4, cutoff_if / (rate / 2), btype="low", output="sos")
    samples_if = sosfiltfilt(sos_if, samples_dc)

    envelope_raw = np.abs(samples_if)

    cutoff_env = 10000
    sos_env = butter(4, cutoff_env / (rate / 2), btype="low", output="sos")
    envelope = sosfiltfilt(sos_env, envelope_raw)

    noise_level = np.median(envelope)
    noise_std = np.std(envelope)
    threshold = (noise_level + 4 * noise_std) / 3

    min_distance = int(200e-6 * rate)
    peaks, _ = find_peaks(envelope, height=threshold, distance=min_distance)

    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([])

    t_peaks = t_start + (peaks / rate)

    time_fraction = t_peaks % 1.0
    cycle_time = time_fraction % 0.02
    phases_detected = (cycle_time / 0.02) * 360.0

    amps_detected = envelope[peaks]

    # dB relatif
    amps_db = 20 * np.log10(np.clip(amps_detected, 1e-12, None))

    # dBm approximatif sur charge 50 ohms
    vrms = amps_detected / np.sqrt(2)
    power_w = (vrms ** 2) / R
    amps_dbm = 10 * np.log10(np.clip(power_w / 1e-3, 1e-20, None))

    return phases_detected, amps_db, amps_dbm


def Plot_PRPD_Multiple(acquisitions, name="", ylabel="Amplitude (dBm)"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["blue", "red", "green", "orange", "purple"]
    all_phases = []
    all_amps = []

    for i, (phases, amps) in enumerate(acquisitions):
        c = colors[i % len(colors)]

        ax1.scatter(
            phases,
            amps,
            s=15,
            c=c,
            alpha=0.6,
            label=f"Acquisition {i + 1}"
        )

        all_phases.extend(phases)
        all_amps.extend(amps)

    phases_ref = np.linspace(0, 360, 500)

    if len(all_amps) > 0:
        y_min = np.min(all_amps)
        y_max = np.max(all_amps)
        y_span = y_max - y_min if y_max > y_min else 40

        wave_50hz = (
            np.sin(np.radians(phases_ref)) * (y_span * 0.4)
            + (y_min + y_span * 0.4)
        )

        ax1.plot(
            phases_ref,
            wave_50hz,
            "-",
            color="black",
            linewidth=2,
            label="Onde 50Hz Référence"
        )

    ax1.set_title(f"Carte PRPD - {name}")
    ax1.set_xlabel("Phase (degrés°)")
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(0, 360)
    ax1.grid(True)
    ax1.legend(loc="lower left")

    if len(all_phases) > 0:
        h = ax2.hist2d(all_phases, all_amps, bins=[128, 50], cmap="jet")
        fig.colorbar(h[3], ax=ax2, label="Densité")

    ax2.set_title(f"Heatmap Globale - {name}")
    ax2.set_xlabel("Phase (degrés°)")
    ax2.set_ylabel(ylabel)
    ax2.set_xlim(0, 360)

    plt.tight_layout()
    plt.savefig(f"./Main_figs/{name}.png", bbox_inches="tight")
    plt.show()


def tx_rx_loopback_sync(usrp, tx_signal, freq, rate):
    max_amp = np.max(np.abs(tx_signal))

    if max_amp > 0:
        tx_signal = 0.5 * tx_signal / max_amp

    usrp.set_tx_rate(rate, 0)
    usrp.set_rx_rate(rate, 0)

    usrp.set_tx_freq(freq, 0)
    usrp.set_rx_freq(freq, 0)

    usrp.set_tx_gain(0, 0)
    usrp.set_rx_gain(20, 0)

    usrp.set_tx_antenna("TX/RX", 0)
    usrp.set_rx_antenna("RX2", 0)

    rx_streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    tx_streamer = usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))

    num_samps = len(tx_signal)
    received = np.zeros(num_samps, dtype=np.complex64)

    rx_md = uhd.types.RXMetadata()
    tx_md = uhd.types.TXMetadata()

    stop_rx = False
    t_start = 0.0

    current_time = usrp.get_time_now().get_real_secs()
    future_time = current_time + 0.1
    start_time = np.ceil(future_time / 0.02) * 0.02

    time_spec = uhd.types.TimeSpec(start_time)

    def rx_worker():
        nonlocal stop_rx, t_start

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = False
        stream_cmd.time_spec = time_spec

        rx_streamer.issue_stream_cmd(stream_cmd)

        total = 0
        buff = np.zeros((1, 4096), dtype=np.complex64)
        first_packet = True

        while total < num_samps and not stop_rx:
            n = rx_streamer.recv(buff, rx_md, timeout=3.0)

            if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
                continue

            if first_packet:
                t_start = rx_md.time_spec.get_real_secs()
                first_packet = False

            end = min(total + n, num_samps)
            received[total:end] = buff[0, :end - total]
            total = end

        rx_streamer.issue_stream_cmd(
            uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        )

    rx_thread = threading.Thread(target=rx_worker)
    rx_thread.start()

    time.sleep(0.1)

    tx_md.start_of_burst = True
    tx_md.end_of_burst = False
    tx_md.has_time_spec = True
    tx_md.time_spec = time_spec

    chunk_size = 4096

    for i in range(0, num_samps, chunk_size):
        chunk = tx_signal[i:i + chunk_size]

        if i + chunk_size >= num_samps:
            tx_md.end_of_burst = True
        # delay = np.random.uniform(0.01, 0.05)
        # delay = 0
        # time.sleep(0.5 + delay)
        tx_streamer.send(chunk, tx_md)

        tx_md.start_of_burst = False
        tx_md.has_time_spec = False

    stop_rx = True
    rx_thread.join()

    return received, t_start


def Time_domain_gr(samples, rate, name=""):
    t = np.arange(len(samples)) / rate
    signal_50 = 0.1 * np.sin(2 * np.pi * 50 * t)

    alpha = int(len(t) / 25)

    plt.figure(figsize=(12, 5))
    plt.plot(t[:alpha], np.real(samples)[:alpha], label="Real part", color="blue")
    plt.plot(t[:alpha], signal_50[:alpha], label="Signal 50 Hz", color="red")

    plt.title("Time Sink")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.savefig(f"./Main_figs/Time_domain_plot{name}.png")
    plt.show()


def Freq_domain_gr_blocks(
    samples,
    rate,
    freq_center,
    nfft=1024,
    f_plot_low=None,
    f_plot_high=None,
    name="freq"
):
    n_blocks = len(samples) // nfft
    samples = samples[:n_blocks * nfft]

    blocks = samples.reshape(n_blocks, nfft)

    window = np.hanning(nfft)
    psd_acc = np.zeros(nfft)

    for blk in blocks:
        X = np.fft.fftshift(np.fft.fft(blk * window, n=nfft))
        P = (np.abs(X) ** 2) / nfft
        psd_acc += P

    psd_mean = psd_acc / n_blocks
    psd_db = 10 * np.log10(psd_mean + 1e-20)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / rate)) + freq_center

    if f_plot_low is not None and f_plot_high is not None:
        mask = (freqs >= f_plot_low) & (freqs <= f_plot_high)
        freqs_plot = freqs[mask]
        psd_plot = psd_db[mask]
    else:
        freqs_plot = freqs
        psd_plot = psd_db

    plt.figure(figsize=(12, 5))
    plt.plot(freqs_plot / 1e6, psd_plot, color="green")

    plt.title("FFT proche du QT GUI Frequency Sink")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Relative Gain (dB)")
    plt.grid()

    plt.savefig(f"./Main_figs/Freq_domain_plot{name}.png")
    plt.show()

    return freqs, psd_db


def main():
    print("Initialisation du SDR avec synchronisation PPS...")

    usrp = uhd.usrp.MultiUSRP()

    usrp.set_time_source("external")

    print("En attente de l'impulsion PPS...")
    time_last = usrp.get_time_last_pps().get_real_secs()

    while True:
        if usrp.get_time_last_pps().get_real_secs() != time_last:
            break
        time.sleep(0.1)

    usrp.set_time_next_pps(uhd.types.TimeSpec(0.0))
    time.sleep(1.2)

    print("Synchronisé !")

    num_samps = int(DURATION * RATE)

    pd_noisy = Simulate_PD_Signal(num_samps, RATE, f_offset=2e6)

    # Time_domain_gr(pd_noisy, RATE, name="_simulated")

    acquisitions_db = []
    acquisitions_dbm = []

    s_time = time.time()

    for i in range(5):
        print(f"--- Acquisition {i + 1}/1 ---")

        delay = np.random.uniform(0.01, 0.05)
        time.sleep(0.5 + delay)

        rx_signal, t_start = tx_rx_loopback_sync(usrp, pd_noisy, FREQ, RATE)

        Time_domain_gr(rx_signal, RATE, name=f"_rx_{i + 1}")

        Freq_domain_gr_blocks(
            rx_signal,
            RATE,
            FREQ,
            NFFT,
            name=f"_rx_{i + 1}"
        )

        phases, amps_db, amps_dbm = Process_PD_Signal(
            rx_signal,
            RATE,
            t_start=t_start,
            f_offset=2e6,
            R=50
        )

        acquisitions_db.append((phases, amps_db))
        acquisitions_dbm.append((phases, amps_dbm))

    e_time = time.time()

    t_time = np.abs(e_time - s_time)
    print("Total time =", t_time)

    # print("Génération du PRPD en dB...")
    # Plot_PRPD_Multiple(
    #     acquisitions_db,
    #     name="Test_AVEC_Synchro_PPS_dB",
    #     ylabel="Amplitude relative (dB)"
    # )

    print("Génération du PRPD en dBm...")
    Plot_PRPD_Multiple(
        acquisitions_dbm,
        name="Test_AVEC_Synchro_PPS_dBm",
        ylabel="Puissance approximative (dBm)"
    )


if __name__ == "__main__":
    main()