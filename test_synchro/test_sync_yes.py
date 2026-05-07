import numpy as np
import matplotlib.pyplot as plt
import uhd
from scipy.signal import butter, find_peaks, sosfiltfilt
import threading
import time

FREQ = 100e6
RATE = 10e6
DURATION = 0.5 # Raccourci pour accélérer les 3 tests

def gaussian_phase(phase_deg, center, sigma):
    return np.exp(-((phase_deg - center)**2) / (2 * sigma**2))

def Simulate_PD_Signal(num_samps, rate, f_offset=10e6):
    t = np.arange(num_samps) / rate
    pd_signal = np.zeros(num_samps, dtype=np.complex64)
    f_ref = 50.0
    phase_ref = (360.0 * f_ref * t) % 360.0
    prob_pd = 0.8 * gaussian_phase(phase_ref, 60, 20) + 1.0 * gaussian_phase(phase_ref, 200, 8)
    prob_pd = prob_pd / np.max(prob_pd)
    
    # IMPORTANT: Graine FIXE. Le signal simulé sera EXACTEMENT le même à chaque appel.
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

    noise = (np.random.normal(0, 0.002, num_samps) + 1j * np.random.normal(0, 0.002, num_samps))
    return pd_signal# + noise

def Process_PD_Signal(samples, rate, t_start=0.0, f_offset=10e6):
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
        return np.array([]), np.array([])

    t_peaks = t_start + (peaks / rate)
    time_fraction = t_peaks % 1.0
    cycle_time = time_fraction % 0.02
    phases_detected = (cycle_time / 0.02) * 360.0

    amps_detected = envelope[peaks]
    amps_db = 20 * np.log10(np.clip(amps_detected, 1e-12, None))

    return phases_detected, amps_db

def Plot_PRPD_Multiple(acquisitions, name=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['blue', 'red', 'green']
    all_phases = []
    all_amps = []

    for i, (phases, amps) in enumerate(acquisitions):
        c = colors[i % len(colors)]
        ax1.scatter(phases, amps, s=15, c=c, alpha=0.6, label=f'Acquisition {i+1}')
        all_phases.extend(phases)
        all_amps.extend(amps)

    phases_ref = np.linspace(0, 360, 500)
    if len(all_amps) > 0:
        y_min, y_max = np.min(all_amps), np.max(all_amps)
        y_span = y_max - y_min if y_max > y_min else 40
        wave_50hz = np.sin(np.radians(phases_ref)) * (y_span * 0.4) + (y_min + y_span * 0.4)
        ax1.plot(phases_ref, wave_50hz, '-', color='black', linewidth=2, label='Onde 50Hz Référence')

    ax1.set_title(f"Carte PRPD - {name}")
    ax1.set_xlabel("Phase (degrés°)")
    ax1.set_ylabel("Amplitude (dBm)")
    ax1.set_xlim(0, 360)
    ax1.grid(True)
    ax1.legend(loc='lower left')

    if len(all_phases) > 0:
        h = ax2.hist2d(all_phases, all_amps, bins=[128, 50], cmap='jet')
        fig.colorbar(h[3], ax=ax2, label="Densité")
    ax2.set_title(f"Heatmap Globale - {name}")
    ax2.set_xlabel("Phase (degrés°)")
    ax2.set_ylabel("Amplitude (dBm)")
    ax2.set_xlim(0, 360)

    plt.tight_layout()
    plt.savefig(f"{name}.png", bbox_inches='tight')
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
    usrp.set_rx_gain(30, 0)
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

    # PLANIFICATION RAPIDE ET ALIGNÉE SUR LE 50HZ (0 délai visible)
    # Au lieu d'attendre la prochaine "seconde ronde" entière (qui peut prendre 2 secondes),
    # on s'aligne sur le prochain "cycle de 50Hz" (0.02s) !
    current_time = usrp.get_time_now().get_real_secs()
    
    # On ajoute juste 100 millisecondes de marge pour la communication USB
    future_time = current_time + 0.1 
    
    # On arrondit au multiple de 0.02s supérieur le plus proche (pour tomber pile sur un 0°)
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
            received[total:end] = buff[0, :end-total]
            total = end

        rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

    rx_thread = threading.Thread(target=rx_worker)
    rx_thread.start()

    time.sleep(0.1)
    tx_md.start_of_burst = True
    tx_md.end_of_burst = False
    tx_md.has_time_spec = True
    tx_md.time_spec = time_spec

    chunk_size = 4096
    for i in range(0, num_samps, chunk_size):
        chunk = tx_signal[i:i+chunk_size]
        if i + chunk_size >= num_samps:
            tx_md.end_of_burst = True
        tx_streamer.send(chunk, tx_md)
        tx_md.start_of_burst = False
        tx_md.has_time_spec = False

    stop_rx = True
    rx_thread.join()
    return received, t_start

def main():
    print("Initialisation du SDR (AVEC SYNCHRONISATION PPS)...")
    usrp = uhd.usrp.MultiUSRP()
    
    usrp.set_time_source("external")
    print("En attente de l'impulsion PPS (Générateur requis)...")
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

    acquisitions = []
    
    # On fait 3 acquisitions successives
    s_time = time.time()
    for i in range(10):
        print(f"--- Acquisition {i+1}/3 ---")
        # delay = np.random.uniform(0.01, 0.05) 
        # time.sleep(0.5 + delay)
        rx_signal, t_start = tx_rx_loopback_sync(usrp, pd_noisy, FREQ, RATE)
        phases, amps = Process_PD_Signal(rx_signal, RATE, t_start=t_start, f_offset=2e6)
        acquisitions.append((phases, amps))
        # time.sleep(0.5)

    e_time = time.time()

    t_time = np.abs(e_time - s_time)
    print("Totale time est = ", t_time)
    print("Génération du graphique...")
    Plot_PRPD_Multiple(acquisitions, "Test_AVEC_Synchro_PPS")

if __name__ == "__main__":
    main()
