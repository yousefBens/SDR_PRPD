import numpy as np
import matplotlib.pyplot as plt
import uhd
from scipy.signal import butter, filtfilt, freqs, find_peaks, sosfiltfilt
import os

import threading
import time


FREQ = 100e6
RATE = 10e6
DURATION = 1.0
GAIN = 50
CHANNEL = 0
NFFT = 1024



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
    
def Freq_domain_gr_blocks(samples, rate, freq_center, nfft=1024, f_plot_low=None, f_plot_high=None, name = "freq"):

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

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/rate)) + freq_center


    if f_plot_low is not None and f_plot_high is not None:
        mask = (freqs >= f_plot_low) & (freqs <= f_plot_high)
        freqs_plot = freqs[mask]
        psd_plot = psd_db[mask]
    else:
        freqs_plot = freqs
        psd_plot = psd_db

    plt.figure(figsize=(12, 5))
    plt.plot(freqs_plot / 1e6, psd_plot)
    plt.title("FFT proche du QT GUI Frequency Sink")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Relative Gain (dB)")
    plt.grid()
    plt.savefig(f"./Main_figs/Freq_domain_plot{name}.png")
    plt.show()

    return freqs, psd_db



def Bandpass_filter_inTimeDomain(samples, f_low, f_hight, rate):
    Wn = [f_low/(rate/2), f_hight/(rate/2)]
    b, a = butter(N=2, Wn=Wn, btype='bandpass')

    filtred_signal = filtfilt(b, a, samples)
    
    return filtred_signal


def Bandpass_complex_inFreqDomain(samples, rate, f_low, f_high):
    N = len(samples)

    # FFT
    X = np.fft.fftshift(np.fft.fft(samples))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/rate))


    mask = (freqs >= f_low) & (freqs <= f_high)

    X_filtered = np.zeros_like(X)
    X_filtered[mask] = X[mask]

    filtered = np.fft.ifft(np.fft.ifftshift(X_filtered))

    return filtered



def gaussian_phase(phase_deg, center, sigma):
    return np.exp(-((phase_deg - center)**2) / (2 * sigma**2))

def Simulate_PD_Signal(num_samps, rate, f_offset=10e6):
    """
    Simule des décharges partielles (PD) sur des phases spécifiques du réseau 50 Hz.
    """
    t = np.arange(num_samps) / rate
    pd_signal = np.zeros(num_samps, dtype=np.complex64)
    
    f_ref = 50.0
    phase_ref = (360.0 * f_ref * t) % 360.0

    # La distribution des DP
    prob_pd = (
        0.8 * gaussian_phase(phase_ref, 60, 20) +
        1.0 * gaussian_phase(phase_ref, 200, 8)
    )
    prob_pd = prob_pd / np.max(prob_pd)
    

    rng = np.random.default_rng(42)
    # On prend 1% * there probs des samples comme des candidats
    p_global = 0.01 * prob_pd
    candidats = np.where(rng.random(num_samps) < p_global)[0]


    # La diff entre les candidats sur les quelle on va appliquer les Pulses de DP
    min_gap_samples = int(200e-6 * rate)
    
    event_indices = []
    last_idx = -min_gap_samples
    for c in candidats:
        if c - last_idx >= min_gap_samples:
            event_indices.append(c)
            last_idx = c
            
    print(f"Nombre d'événements PD simulés : {len(event_indices)}")


    pulse_duration = 150e-6
    pulse_len = int(pulse_duration * rate)
    tau = 30e-6
    tp = np.arange(pulse_len) / rate
    
    for idx in event_indices:
        A = rng.uniform(0.0015, 0.041)

        pulse = A * np.exp(-tp / tau) * np.exp(1j * 2 * np.pi * f_offset * tp)
        
        end_idx = min(idx + pulse_len, num_samps)
        valid_len = end_idx - idx
        pd_signal[idx:end_idx] += pulse[:valid_len]

    return pd_signal

def Process_PD_Signal(samples, rate, t_start=0.0, f_offset=10e6):
    """
    Traite le signal reçu SDR pour générer un PRPD propre :
    1) translation vers 0 Hz
    2) filtrage IF
    3) extraction enveloppe
    4) détection des impulsions PD
    5) conversion pic -> phase 50 Hz + amplitude
    """

    N = len(samples)
    t = np.arange(N) / rate

    print(" -> Translation en bande de base...")
    samples_dc = samples * np.exp(-1j * 2 * np.pi * f_offset * t)

    print(" -> Filtrage IF autour de 0 Hz...")
    cutoff_if = 1e6
    sos_if = butter(
        4,
        cutoff_if / (rate / 2),
        btype="low",
        output="sos"
    )
    samples_if = sosfiltfilt(sos_if, samples_dc)

    print(" -> Extraction de l'enveloppe...")
    envelope_raw = np.abs(samples_if)

    print(" -> Lissage de l'enveloppe...")
    cutoff_env = 10000  # 10 kHz, mieux pour garder les impulsions rapides
    sos_env = butter(
        4,
        cutoff_env / (rate / 2),
        btype="low",
        output="sos"
    )
    envelope = sosfiltfilt(sos_env, envelope_raw)

    print(" -> Détection des pics PD...")

    noise_level = np.median(envelope)
    noise_std = np.std(envelope)

    threshold = noise_level + 4 * noise_std
    
    threshold = threshold/3
    print("Threshold == ", threshold)

    min_distance = int(200e-6 * rate)

    peaks, properties = find_peaks(
        envelope,
        height=threshold,
        distance=min_distance
    )

    print(f" -> Nombre de pics détectés : {len(peaks)}")

    if len(peaks) == 0:
        return np.array([]), np.array([])

    f_ref = 50.0

    # --- NOUVEAU CALCUL DE PHASE AVEC PPS ---
    # Le temps absolu de chaque pic
    t_peaks = t_start + (peaks / rate)
    
    # On isole la fraction de seconde (ex: 1.145 -> 0.145)
    time_fraction = t_peaks % 1.0
    
    # On trouve la position dans le cycle de 20 ms
    cycle_time = time_fraction % 0.02
    
    # Conversion en degrés
    phases_detected = (cycle_time / 0.02) * 360.0

    amps_detected = envelope[peaks]

    amps_db = 20 * np.log10(np.clip(amps_detected, 1e-12, None))

    print(f" -> Terminé. Nombre total de points PRPD : {len(phases_detected)}")

    return phases_detected, amps_db




def Plot_PRPD(phases_detected, amps_dbm, name=""):
    """
    Affiche la matrice PRPD comme sur un équipement UHF100 (Tous les points tracés).
    """
    os.makedirs("./Main_figs", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


    ax1.scatter(phases_detected, amps_dbm, s=5, c='blue', alpha=0.15, label='Mesures (Bruit + DP)')
    

    phases_ref = np.linspace(0, 360, 500)
    y_min = np.min(amps_dbm)
    y_max = np.max(amps_dbm)
    y_span = y_max - y_min if y_max > y_min else 40
    
    wave_50hz = np.sin(np.radians(phases_ref)) * (y_span * 0.4) + (y_min + y_span * 0.4)
    ax1.plot(phases_ref, wave_50hz, '-', color='black', linewidth=2, label='Onde 50Hz')
    
    ax1.set_title(f"Carte PRPD (Nuage de points dense)")
    ax1.set_xlabel("Phase (degrés°)")
    ax1.set_ylabel("Amplitude (dBm)")
    ax1.set_xlim(0, 360)
    ax1.grid(True)
    ax1.legend(loc='lower left')


    h = ax2.hist2d(phases_detected, amps_dbm, bins=[256, 50], cmap='jet')
    fig.colorbar(h[3], ax=ax2, label="Densité de points")
    ax2.set_title("PRPD Heatmap 2D")
    ax2.set_xlabel("Phase (degrés°)")
    ax2.set_ylabel("Amplitude (dBm)")
    ax2.set_xlim(0, 360)

    plt.tight_layout()
    plt.savefig(f"./Main_figs/PRPD_plot{name}.png", bbox_inches='tight')
    plt.show()



def tx_rx_loopback(usrp, tx_signal, freq, rate, tx_gain=0, rx_gain=20, tx_chan=0, rx_chan=0):

    tx_signal = tx_signal.astype(np.complex64)

    max_amp = np.max(np.abs(tx_signal))
    if max_amp > 0:
        tx_signal = 0.5 * tx_signal / max_amp

    usrp.set_tx_rate(rate, tx_chan)
    usrp.set_rx_rate(rate, rx_chan)

    usrp.set_tx_freq(freq, tx_chan)
    usrp.set_rx_freq(freq, rx_chan)

    usrp.set_tx_gain(tx_gain, tx_chan)
    usrp.set_rx_gain(rx_gain, rx_chan)

    usrp.set_tx_antenna("TX/RX", tx_chan)
    usrp.set_rx_antenna("RX2", rx_chan)

    st_args_rx = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args_rx.channels = [rx_chan]
    rx_streamer = usrp.get_rx_stream(st_args_rx)

    st_args_tx = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args_tx.channels = [tx_chan]
    tx_streamer = usrp.get_tx_stream(st_args_tx)

    num_samps = len(tx_signal)
    received = np.zeros(num_samps, dtype=np.complex64)

    rx_md = uhd.types.RXMetadata()
    tx_md = uhd.types.TXMetadata()

    stop_rx = False
    t_start = 0.0

    def rx_worker():
        nonlocal stop_rx, t_start

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        rx_streamer.issue_stream_cmd(stream_cmd)

        total = 0
        buff = np.zeros((1, 4096), dtype=np.complex64)
        first_packet = True

        while total < num_samps and not stop_rx:
            n = rx_streamer.recv(buff, rx_md, timeout=2.0)

            if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
                print("RX error:", rx_md.strerror())
                continue
                
            if first_packet:
                t_start = rx_md.time_spec.get_real_secs()
                first_packet = False

            end = min(total + n, num_samps)
            received[total:end] = buff[0, :end-total]
            total = end

        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx_streamer.issue_stream_cmd(stop_cmd)

        print(f"RX reçu : {total} samples (Timestamp initial absolu: {t_start:.6f} s)")

    rx_thread = threading.Thread(target=rx_worker)
    rx_thread.start()

    time.sleep(0.1)

    tx_md.start_of_burst = True
    tx_md.end_of_burst = False
    tx_md.has_time_spec = False

    chunk_size = 4096
    sent_total = 0

    for i in range(0, num_samps, chunk_size):
        chunk = tx_signal[i:i+chunk_size]

        if i + chunk_size >= num_samps:
            tx_md.end_of_burst = True

        sent = tx_streamer.send(chunk, tx_md)
        sent_total += sent
        tx_md.start_of_burst = False

    stop_rx = True
    rx_thread.join()

    print(f"TX envoyé : {sent_total} samples")

    return received, t_start


def Find_PD_Band(samples, rate, band_width=1e6, step=500e3):
    """
    Cherche automatiquement la bande fréquentielle où les impulsions PD sont les plus visibles.
    Retourne le meilleur f_offset.
    """

    offsets = np.arange(-rate/2 + band_width, rate/2 - band_width, step)

    best_score = -np.inf
    best_offset = None

    for f_offset in offsets:
        N = len(samples)
        t = np.arange(N) / rate

        # Translation de la bande testée vers 0 Hz
        x = samples * np.exp(-1j * 2 * np.pi * f_offset * t)

        # Filtre passe-bas sur la bande testée
        cutoff = band_width / 2
        sos = butter(
            4,
            cutoff / (rate / 2),
            btype="low",
            output="sos"
        )

        x_filt = sosfiltfilt(sos, x)

        # Enveloppe
        env = np.abs(x_filt)

        # Score impulsionnel
        noise = np.median(env)
        sigma = np.std(env)
        threshold = noise + 4 * sigma

        peaks, _ = find_peaks(
            env,
            height=threshold,
            distance=int(10e-6 * rate)
        )
        print("Len picks == ", len(peaks))
        if len(peaks) == 0:
            score = 0
        else:
            score = len(peaks) * np.mean(env[peaks]) / (noise + 1e-12)

        if score > best_score:
            best_score = score
            best_offset = f_offset

    print(f"Meilleure bande trouvée autour de f_offset = {best_offset/1e6:.3f} MHz")
    print(f"Score = {best_score:.2f}")

    return best_offset

def main():
    print("Initializing USRP...")
    usrp = uhd.usrp.MultiUSRP()

    print("\n--- 1. SYNCHRONISATION PPS ---")
    usrp.set_time_source("external")
    print("En attente de la première impulsion (1 Hz) de votre générateur...")
    print("=> ALLUMEZ LE GÉNÉRATEUR CONNECTÉ SUR PPS/TRIG ! <=")
    time_last = usrp.get_time_last_pps().get_real_secs()
    while True:
        time_curr = usrp.get_time_last_pps().get_real_secs()
        if time_curr != time_last:
            print("   -> Impulsion PPS détectée !")
            break
        time.sleep(0.1)
    
    usrp.set_time_next_pps(uhd.types.TimeSpec(0.0))
    time.sleep(1.2)
    print(f"Synchronisation réussie ! Temps actuel SDR : {usrp.get_time_now().get_real_secs():.4f} s")

    num_samps = int(DURATION * RATE)

    print("\n--- SIMULATION PD ---")
    F_OFFSET = 2e6

    pd_simulated = Simulate_PD_Signal(
        num_samps,
        RATE,
        f_offset=F_OFFSET
    )

    noise_power = 0.0025
    
    noise = (
        np.random.normal(0, noise_power, len(pd_simulated)) +
        1j * np.random.normal(0, noise_power, len(pd_simulated))
    )
    
    # pd_noisy = pd_simulated + noise
    f_noise = 4e6
    
    t = np.arange(len(pd_simulated)) / RATE
    
    rf_noise = 0.002 * np.exp(1j * 2 * np.pi * f_noise * t)
    
    pd_noisy = pd_simulated + rf_noise + noise
    drift = 0.001 * np.sin(2*np.pi*5*t)
    
    pd_noisy += drift
    # pd_simulated = pd_simulated * 0

    print("\n--- TX puis RX ---")
    rx_signal, t_start = tx_rx_loopback(
        usrp=usrp,
        tx_signal=pd_noisy,
        freq=FREQ,
        rate=RATE,
        tx_gain=0,
        rx_gain=30,
        tx_chan=0,
        rx_chan=0
    )
    
    # best_offset = Find_PD_Band(rx_signal, RATE, step=300e3)

    # print("Best offsets ====== ", best_offset)
    
    # phases_detected, amps_dbm = Process_PD_Signal(
    #     rx_signal,
    #     RATE,
    #     f_offset=best_offset
    # )
    
    print("\n--- TRAITEMENT PRPD SUR SIGNAL REÇU ---")
    phases_detected, amps_dbm = Process_PD_Signal(
        rx_signal,
        RATE,
        t_start=t_start,
        f_offset=F_OFFSET
    )

    if len(amps_dbm) > 0:
        Plot_PRPD(phases_detected, amps_dbm, name="_tx_rx_b200")
    else:
        print("Pas assez de samples RX pour tracer le PRPD.")

    Freq_domain_gr_blocks(
        rx_signal,
        RATE,
        FREQ,
        NFFT,
        name="_rx_signal"
    )

    Time_domain_gr(pd_noisy, RATE)

    

if __name__ == "__main__":
    main()