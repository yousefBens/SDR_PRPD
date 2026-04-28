import numpy as np
import matplotlib.pyplot as plt
import uhd
from scipy.signal import butter, filtfilt, freqs, find_peaks, sosfiltfilt
import os



FREQ = 100e6
RATE = 32e6
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
        0.8 * gaussian_phase(phase_ref, 60, 5) +
        1.0 * gaussian_phase(phase_ref, 240, 15)
    )
    prob_pd = prob_pd / np.max(prob_pd)
    

    rng = np.random.default_rng(42)
    # On prend 1% * there probs des samples comme des candidats
    p_global = 0.0001 * prob_pd
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

def Process_PD_Signal(samples, rate, f_offset=10e6):
    """
    Traite le signal brut SDR.
    """
    N = len(samples)
    t = np.arange(N) / rate
    
    print(" -> Translation en Bande de Base (-10 MHz)...")
    samples_dc = samples * np.exp(-1j * 2 * np.pi * f_offset * t)
    
    print(" -> Filtrage IF (Bande étroite UHF100 : 3 MHz)...")
 
    sos_if = butter(4, 1e6 / (rate / 2), btype='low', output='sos')
    samples_if = sosfiltfilt(sos_if, samples_dc)
    
    print(" -> Extraction de l'enveloppe brute...")
    envelope_raw = np.abs(samples_if)
    
    print(" -> Filtrage Passe-Bas de l'enveloppe (3000 Hz)...")

    sos_env = butter(4, 3000 / (rate / 2), btype='low', output='sos')
    envelope = sosfiltfilt(sos_env, envelope_raw)
    
    print(" -> Formatage du PRPD (256 échantillons/période)...")
    f_ref = 50.0
    samples_per_period = int(rate / f_ref)
    N_periods = N // samples_per_period
    
    phases_detected = []
    amps_detected = []
    
    for p in range(N_periods):
        start = p * samples_per_period
        end = start + samples_per_period
        period_env = envelope[start:end]
        

        chunks = np.array_split(period_env, 256)
        chunk_maxes = [np.max(chunk) for chunk in chunks]
        amps_detected.extend(chunk_maxes)

    phases_detected = np.tile(np.linspace(0, 360, 256, endpoint=False), N_periods)
    amps_detected = np.array(amps_detected)
    


    amps_dbm = 20 * np.log10(np.clip(amps_detected, 1e-12, None))
    
    print(f" -> Terminé. Nombre total de points générés pour le PRPD : {len(phases_detected)}")
    
    return phases_detected, amps_dbm

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


def main():
    f_low = 10e6 - 1000000
    f_hight = 10e6 + 1000000
    print("Initializing USRP...")

    usrp = uhd.usrp.MultiUSRP()

    num_samps = int(DURATION * RATE)

    print(f"Receiving {num_samps} samples...")
    samples = usrp.recv_num_samps(
        num_samps,
        FREQ,
        RATE,
        [CHANNEL],
        GAIN
    )

    if samples.ndim == 2:
        samples = samples[0]

    print(f"Received {len(samples)} samples")
    print(type(samples))


    print("\n--- DEBUT: SIMULATION & TRAITEMENT DP ---")
    F_OFFSET = 10e6
    

    print("Génération de Décharges Partielles simulées...")
    pd_simulated = Simulate_PD_Signal(len(samples), RATE, f_offset=F_OFFSET)
    

    samples_with_pd = samples + pd_simulated

    phases_detected, amps_dbm = Process_PD_Signal(samples_with_pd, RATE, f_offset=F_OFFSET)

    Plot_PRPD(phases_detected, amps_dbm, name="_simulation_b200")
    print("--- FIN: SIMULATION & TRAITEMENT DP ---\n")
    

    # # Plot signal Brute
    # freqs, psd_db = Freq_domain_gr_blocks(samples, RATE, FREQ, NFFT, f_plot_low=None, f_plot_high=None, name = "")
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