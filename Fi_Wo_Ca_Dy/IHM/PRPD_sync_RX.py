import numpy as np
import matplotlib.pyplot as plt
import uhd
from scipy.signal import butter, find_peaks, sosfiltfilt
import time
import os



NFFT = 32768
FREQ = 1.196e9
RATE = 12e6
DURATION = 3
GAIN = 40
CHANNEL = 0
ANTENNA = "RX2"

EDGE = (RATE/2)*0.3


R = 50


F_OFFSET = 0e6

N_ACQ = 1

OUTPUT_DIR = "./Main_figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)




import numpy as np

def compute_spectrum_dbfs(
        samples,
        rate,
        freq_center,
        nfft=4096,
        remove_dc=True,
        edge_guard_hz=0.5e6
):
    samples = np.asarray(samples, dtype=np.complex64).ravel()

    if len(samples) < nfft:
        return np.array([]), np.array([])


    if remove_dc:
        samples = samples - np.mean(samples)


    n_blocks = len(samples) // nfft
    samples = samples[:n_blocks * nfft]

    blocks = samples.reshape(n_blocks, nfft)


    window = np.hanning(nfft)


    coherent_gain = np.sum(window) / nfft


    p_acc = np.zeros(nfft)

    for blk in blocks:


        xw = blk * window


        X = np.fft.fftshift(np.fft.fft(xw, nfft))


        X = X / (nfft * coherent_gain)


        power = np.abs(X) ** 2


        p_acc += power


    p_mean = p_acc / n_blocks

    full_scale_power = 1.0


    spectrum_dbfs = 10 * np.log10(
        np.clip(p_mean / full_scale_power, 1e-20, None)
    )


    freqs = np.fft.fftshift(
        np.fft.fftfreq(nfft, d=1 / rate)
    ) + freq_center


    half_bw = rate / 2

    mask_edges = (
        (freqs >= freq_center - half_bw + edge_guard_hz) &
        (freqs <= freq_center + half_bw - edge_guard_hz)
    )

    freqs = freqs[mask_edges]
    spectrum_dbfs = spectrum_dbfs[mask_edges]

    return freqs, spectrum_dbfs

def plot_spectrum(freqs, psd_dbm, name="Freq_domain_plot", unity = "dBm/bin"):
    if len(freqs) == 0:
        print("Spectre vide.")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(freqs / 1e6, psd_dbm)

    plt.title(f"Spectre FFT - Puissance approximative en {unity}")
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel(f"Puissance approximative ({unity})")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def rx_only_sync(usrp, freq, rate, duration, gain=0, antenna="RX2"):
    num_samps = int(duration * rate)

    usrp.set_rx_rate(rate, CHANNEL)
    usrp.set_rx_freq(uhd.types.TuneRequest(freq), CHANNEL)
    usrp.set_rx_gain(gain, CHANNEL)
    usrp.set_rx_antenna(antenna, CHANNEL)

    usrp.set_rx_dc_offset(True, CHANNEL)
    usrp.set_rx_iq_balance(True, CHANNEL)

    time.sleep(0.5)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [CHANNEL]

    rx_streamer = usrp.get_rx_stream(stream_args)
    rx_md = uhd.types.RXMetadata()

    received = np.zeros(num_samps, dtype=np.complex64)

    current_time = usrp.get_time_now().get_real_secs()
    future_time = current_time + 0.1

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




def process_pd_signal_dbfs(samples, rate, t_start=0.0, f_offset=5e6):
    samples = np.asarray(samples, dtype=np.complex64).ravel()
    N = len(samples)

    if N == 0 or t_start is None:
        return np.array([]), np.array([])

    samples = samples - np.mean(samples)

    t = np.arange(N) / rate

    # Translation vers DC
    samples_bb = samples * np.exp(-1j * 2 * np.pi * f_offset * t)

    # Filtrage autour du signal PD
    cutoff_if = 5e6
    sos_if = butter(
        4,
        cutoff_if / (rate / 2),
        btype="low",
        output="sos"
    )

    samples_if = sosfiltfilt(sos_if, samples_bb)

    # Enveloppe
    envelope = np.abs(samples_if)

    # Seuil robuste
    noise_level = np.median(envelope)
    noise_std = np.std(envelope)
    threshold = noise_level + 4.0 * noise_std*1/2

    min_distance = int(200e-6 * rate)

    peaks, props = find_peaks(
        envelope,
        height=threshold,
        distance=min_distance
    )

    if len(peaks) == 0:
        return np.array([]), np.array([])

    t_peaks = t_start + peaks / rate

    # PRPD 50 Hz
    phases = ((t_peaks % 0.02) / 0.02) * 360.0

    amps = envelope[peaks]

    amps_dbfs = 20 * np.log10(np.clip(amps, 1e-12, None))

    return phases, amps_dbfs
    
def plot_prpd(acquisitions, name="PRPD_dBm", unity = "dbm"):
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
    ax1.set_ylabel(f"Puissance impulsion ({unity})")
    ax1.set_xlim(0, 360)
    # ax1.set_ylim(-100, -30)
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
    ax2.set_ylabel("Puissance impulsion (dBfs)")
    ax2.set_xlim(0, 360)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", bbox_inches="tight")
    plt.show()
def sync_on_external_50hz_pps(usrp):
    usrp.set_clock_source("internal")
    usrp.set_time_source("external")

    time_last = usrp.get_time_last_pps().get_real_secs()
    while usrp.get_time_last_pps().get_real_secs() == time_last:
        time.sleep(0.001)

    usrp.set_time_next_pps(uhd.types.TimeSpec(0.0))
    time.sleep(0.05) # Le temps que le prochain front passe et remette à zéro


def Time_domain_gr(samples, rate, name = ""):
    t = np.arange(0, DURATION, 1/rate)
    signal_50 = 0.1 * np.sin(2*np.pi*50*t)
    alpha = int(len(t)/40)
    plt.figure(figsize=(12, 5))
    plt.plot(t[:alpha], np.real(samples)[:alpha], label = "Real part")
    # plt.plot(t[:alpha], np.imag(samples)[:alpha], label = "Imag part")
    plt.plot(t[:alpha], signal_50[:alpha], label = "Signal 50 Hz")
    plt.title("Time Sink")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.legend()
    plt.grid()
    plt.savefig(f"./Main_figs/Time_domain_plot{name}.png")
    plt.show()


def main():
    print("Initialisation USRP...")

    usrp = uhd.usrp.MultiUSRP()

    acquisitions_dbm = []
    acquisitions_dBFS = []

    t0 = time.time()

    for i in range(N_ACQ):
        print(f"\n--- Acquisition {i + 1}/{N_ACQ} ---")

        # Il est très important de se re-synchroniser AVANT CHAQUE ACQUISITION.
        # Sinon, l'horloge du SDR dérive lentement et la phase tournera au fil des minutes !
        print("Synchronisation PPS externe sur le 50Hz...")
        sync_on_external_50hz_pps(usrp)

        rx_signal, t_start = rx_only_sync(
            usrp=usrp,
            freq=FREQ,
            rate=RATE,
            duration=DURATION,
            gain=GAIN,
            antenna=ANTENNA
        )

        if len(rx_signal) == 0:
            print("Acquisition vide.")
            continue

        Time_domain_gr(rx_signal, RATE, name = "")
        # freqs, psd_dbm = compute_spectrum_dbm_per_bin(
        #     rx_signal,
        #     RATE,
        #     FREQ,
        #     nfft=NFFT,
        #     R=R,
        #     remove_dc=True,
        #     edge_guard_hz=EDGE
        # )

        freqss, psd_dBFS = compute_spectrum_dbfs(
            rx_signal,
            RATE,
            FREQ,
            nfft=NFFT,
            remove_dc=True,
            edge_guard_hz=EDGE
        )
        # np.savez(
        #     "/home/yousef/Documents/testing_scripts/GEVernova/After_Aix/CAL2B_files/spectrum_None2_DP.npz",
        #     freqs=freqss,
        #     psd=psd_dBFS
        # )
        np.savez(
            "/home/yousef/Documents/testing_scripts/GEVernova/After_Aix/CAL2B_files/Time_WithH_DP.npz",
            ampl = rx_signal
        )
        
        # plot_spectrum(
        #     freqs,
        #     psd_dbm,
        #     name=f"Spectrum_dBm_bin_rx_{i + 1}"
        # )
        
        plot_spectrum(
            freqss,
            psd_dBFS,
            name=f"Spectrum_dBFS_bin_rx_{i + 1}",
            unity = "dBFS"
        )
        
        # phases, amps_dbm = process_pd_signal_dbm(
        #     rx_signal,
        #     RATE,
        #     t_start=t_start,
        #     f_offset=F_OFFSET,
        #     R=R
        # )
        phasess, amps_dBFS = process_pd_signal_dbfs(
            rx_signal,
            RATE,
            t_start=t_start,
            f_offset=F_OFFSET
        )

        # print(f"Nombre de pulses détectés : {len(phases)}")

        # acquisitions_dbm.append((phases, amps_dbm))
        acquisitions_dBFS.append((phasess, amps_dBFS))

    print(f"\nTemps total = {time.time() - t0:.2f} s")

    # plot_prpd(
    #     acquisitions_dbm,
    #     name="PRPD_dBm_approx"
    # )
    plot_prpd(
        acquisitions_dBFS,
        name="PRPD_dBFS_approx",
        unity = "dbFS"
    )

if __name__ == "__main__":
    main()