import numpy as np
import matplotlib.pyplot as plt
import uhd
from scipy.signal import butter, filtfilt, freqs, find_peaks
import os


# =======================
# CONFIGURATION
# =======================
FREQ = 100e6
# RATE = 30e6
# DURATION = 1.0
GAIN = 40
CHANNEL0 = 0
CHANNEL1 = 1
# NFFT = 1024
RATE = 10e6
NFFT = 1024
DURATION = 1.0

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
    # garder un multiple entier de nfft
    n_blocks = len(samples) // nfft
    samples = samples[:n_blocks * nfft]

    # découpage en blocs
    blocks = samples.reshape(n_blocks, nfft)

    # fenêtre comme un analyseur
    window = np.hanning(nfft)

    # accumulation PSD
    psd_acc = np.zeros(nfft)

    for blk in blocks:
        X = np.fft.fftshift(np.fft.fft(blk * window, n=nfft))
        P = (np.abs(X) ** 2) / nfft
        psd_acc += P

    psd_mean = psd_acc / n_blocks
    psd_db = 10 * np.log10(psd_mean + 1e-20)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/rate)) + freq_center

    # interval choice
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

    # masque : garder uniquement fréquences positives
    mask = (freqs >= f_low) & (freqs <= f_high)

    # appliquer masque
    X_filtered = np.zeros_like(X)
    X_filtered[mask] = X[mask]

    # retour temps
    filtered = np.fft.ifft(np.fft.ifftshift(X_filtered))

    return filtered

# ==================================
# MODULE DECHARGES PARTIELLES (PRPD)
# ==================================

def gaussian_phase(phase_deg, center, sigma):
    return np.exp(-((phase_deg - center)**2) / (2 * sigma**2))

def Simulate_PD_Signal(num_samps, rate, f_offset=10e6):
    """
    Simule des décharges partielles (PD) sur des phases spécifiques du réseau 50 Hz.
    Basé sur les probabilités du notebook (Sim_PRPD.ipynb).
    """
    t = np.arange(num_samps) / rate
    pd_signal = np.zeros(num_samps, dtype=np.complex64)
    
    f_ref = 50.0
    phase_ref = (360.0 * f_ref * t) % 360.0
    
    prob_pd = (
        0.8 * gaussian_phase(phase_ref, 60, 12) +
        0.7 * gaussian_phase(phase_ref, 240, 15)
    )
    prob_pd = prob_pd / np.max(prob_pd)
    
    # 3. Génération des événements (Vectorisée pour fonctionner vite à 32Msps)
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
            
    print(f"Nombre d'événements PD simulés : {len(event_indices)}")

    # 4. Ajout des pulses
    pulse_duration = 150e-6
    pulse_len = int(pulse_duration * rate)
    tau = 30e-6
    tp = np.arange(pulse_len) / rate
    
    for idx in event_indices:
        A = rng.uniform(0.005, 0.04)
        # Utilisation de f_offset pour décaler la PD en IQ Baseband (SDR)
        pulse = A * np.exp(-tp / tau) * np.exp(1j * 2 * np.pi * f_offset * tp)
        
        end_idx = min(idx + pulse_len, num_samps)
        valid_len = end_idx - idx
        pd_signal[idx:end_idx] += pulse[:valid_len]

    return pd_signal

def Process_PD_Signal(samples, rate, f_offset=10e6):
    """
    Traite le signal brut SDR, basé sur Sim_PRPD.
    """
    N = len(samples)
    t = np.arange(N) / rate
    
    print(" -> Translation en Bande de Base (-10 MHz)...")
    samples_dc = samples * np.exp(-1j * 2 * np.pi * f_offset * t)
    
    print(" -> Extraction de l'enveloppe brute...")
    envelope_raw = np.abs(samples_dc)
    
    print(" -> Filtrage Passe-Bas de l'enveloppe (3000 Hz)...")
    b, a = butter(4, 3000 / (rate / 2), btype='low')
    envelope = filtfilt(b, a, envelope_raw)
    
    print(" -> Détection des impulsions...")
    threshold = np.mean(envelope) + 2 * np.std(envelope)
    min_distance = int(0.0004 * rate)   # 0.4 ms
    
    peaks, properties = find_peaks(envelope, height=threshold, distance=min_distance)
    print(f" -> Terminé. Nombre de décharges détectées : {len(peaks)}")
    
    return envelope, peaks, properties

def Plot_PRPD(peaks, properties, rate, num_samps, name=""):
    """
    Affiche la matrice PRPD en utilisant la méthode des Zero Crossings.
    """
    if len(peaks) == 0:
        print("Aucun pic n'a été détecté pour générer un PRPD.")
        return
        
    t = np.arange(num_samps) / rate
    f_ref = 50.0
    ref50 = np.sin(2 * np.pi * f_ref * t)
    
    zc_indices = np.where((ref50[:-1] < 0) & (ref50[1:] >= 0))[0]
    zc_times = zc_indices / rate
    
    phases_detected = []
    amps_detected = []
    
    for peak_idx, amp in zip(peaks, properties['peak_heights']):
        t_imp = peak_idx / rate
        
        k = np.searchsorted(zc_times, t_imp) - 1
        if k >= 0 and k < len(zc_times) - 1:
            t0 = zc_times[k]
            t1 = zc_times[k + 1]
            
            phase = 360.0 * (t_imp - t0) / (t1 - t0)
            phases_detected.append(phase)
            amps_detected.append(amp)

    os.makedirs("./Main_figs", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Graphe Scatter
    ax1.scatter(phases_detected, amps_detected, s=12, c='red', alpha=0.6, label='Décharges')
    phases_ref = np.linspace(0, 360, 500)
    amplitude_max = np.max(amps_detected) * 1.1 if len(amps_detected) > 0 else 1.0
    wave_50hz = np.sin(np.radians(phases_ref)) * amplitude_max
    ax1.plot(phases_ref, wave_50hz, '--', color='gray', alpha=0.3, label='Référence AC')
    ax1.set_title(f"Carte PRPD Simulée")
    ax1.set_xlabel("Phase (degrés)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, 360)
    ax1.grid(True)
    ax1.legend()

    # Graphe Histogramme 2D
    h = ax2.hist2d(phases_detected, amps_detected, bins=[72, 50], cmap='viridis')
    fig.colorbar(h[3], ax=ax2, label="Nombre d'occurrences")
    ax2.set_title("PRPD 2D (Histogramme)")
    ax2.set_xlabel("Phase (degrés)")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim(0, 360)

    plt.tight_layout()
    plt.savefig(f"./Main_figs/PRPD_plot{name}.png", bbox_inches='tight')
    plt.show()

# =======================
# MAIN
# =======================
def main():
    f_low = 10e6 - 1000000
    f_hight = 10e6 + 1000000
    print("Initializing USRP...")

    usrp = uhd.usrp.MultiUSRP()

    num_samps = int(DURATION * RATE)

    print(f"Receiving {num_samps} samples...")
    samples_2dim = usrp.recv_num_samps(
        num_samps,
        FREQ,
        RATE,
        [CHANNEL0, CHANNEL1],
        GAIN
    )

    print("Sample ndim ========= ", samples_2dim.ndim)
    if samples_2dim.ndim == 2:
        samples = samples_2dim[0]
        samples_p = samples_2dim[1]

    print(f"Received {len(samples)} samples")
    print(type(samples))

    # ========================================================
    # INJECTION ET SYNCHRONISATION PRPD (NOUVEAU)
    # ========================================================
    # print("\n--- DEBUT: SIMULATION & TRAITEMENT DP ---")
    # F_OFFSET = 5e6
    
    # # 1. On génère numériquement un vecteur signal de défaut (DP)
    # print("Génération de Décharges Partielles simulées...")
    # pd_simulated = Simulate_PD_Signal(len(samples), RATE, f_offset=F_OFFSET)
    
    # # 2. On ajoute ça directement aux "samples" réels reçus de l'antenne SDR
    # samples_with_pd = samples + pd_simulated
    
    # # 3. DSP (Digital Signal Processing) de l'extraction
    # envelope, peaks, properties = Process_PD_Signal(samples_with_pd, RATE, f_offset=F_OFFSET)
    
    # # 4. Affichage du Diagramme PRPD
    # Plot_PRPD(peaks, properties, RATE, len(samples_with_pd), name="_simulation_b200")
    # print("--- FIN: SIMULATION & TRAITEMENT DP ---\n")
    

    # Plot signal Brute
    # freqs, psd_db = Freq_domain_gr_blocks(samples_with_pd, RATE, FREQ, NFFT, f_plot_low=None, f_plot_high=None, name = "")
    _, _ = Freq_domain_gr_blocks(samples, RATE, FREQ, NFFT, f_plot_low=None, f_plot_high=None, name = "_First_Channel_without50HZ")
    # _, _ = Freq_domain_gr_blocks(pd_simulated, RATE, FREQ, NFFT, f_plot_low=None, f_plot_high=None, name = "")
    # _, _ = Freq_domain_gr_blocks(samples_with_pd, RATE, FREQ, NFFT, f_plot_low=None, f_plot_high=None, name = "")
    # Time_domain_gr(pd_simulated, RATE)
    
    # # Band pass in time
    # filtred_signal = Bandpass_filter_inTimeDomain(samples, f_low, f_hight, RATE)
    # Time_domain_gr(filtred_signal, RATE, name = "_bandpass")

    # # Plot bandpass time signal
    # freqs_filter, psd_db_filter = Freq_domain_gr_blocks(filtred_signal, RATE, FREQ, NFFT, f_plot_low=f_low+FREQ, f_plot_high=f_hight+FREQ, name = "_bandpass")

    # # band pass in freq
    # filtred_signal_by_freq = Bandpass_complex_inFreqDomain(samples, RATE, f_low, f_hight)
    # Time_domain_gr(filtred_signal_by_freq, RATE, name = "_bandpass_by_freq")

    # # Plot bandpass frq Signal
    # freqs_filter_by_freq, psd_db_filter_by_freq = Freq_domain_gr_blocks(filtred_signal_by_freq, RATE, FREQ, NFFT, f_plot_low=f_low+FREQ, f_plot_high=f_hight+FREQ, name = "_bandpass_by_freq")
    

if __name__ == "__main__":
    main()