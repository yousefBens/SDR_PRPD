import numpy as np
import matplotlib.pyplot as plt
import uhd
from scipy.signal import butter, find_peaks, sosfiltfilt
import threading
import time
import os

# =========================
# CONFIGURATION
# =========================

NFFT = 4096
FREQ = 1e9 - 2e6
RATE = 30e6
DURATION = 0.5

TX_GAIN = 0
RX_GAIN = 20

CHANNEL = 0
TX_ANTENNA = "TX/RX"
RX_ANTENNA = "RX2"

EDGE = (RATE / 2) * 0.3

F_OFFSET = 2e6
R = 50
N_ACQ = 5

OUTPUT_DIR = "./Main_figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# SYNCHRO EXTERNE 50 Hz VIA PPS
# =========================

def sync_on_external_50hz_pps(usrp):
    usrp.set_clock_source("internal")
    usrp.set_time_source("external")

    print("Attente front externe 50 Hz sur PPS IN...")

    last_pps = usrp.get_time_last_pps().get_real_secs()

    while usrp.get_time_last_pps().get_real_secs() == last_pps:
        time.sleep(0.001)

    usrp.set_time_next_pps(uhd.types.TimeSpec(0.0))
    time.sleep(0.05)

    print("Temps USRP remis à zéro sur front externe 50 Hz.")


# =========================
# SIMULATION SIGNAL PRPD
# =========================

def gaussian_phase(phase_deg, center, sigma):
    return np.exp(-((phase_deg - center) ** 2) / (2 * sigma ** 2))


def simulate_pd_signal(num_samps, rate, f_offset=2e6, random_phase=True):
    t = np.arange(num_samps) / rate
    pd_signal = np.zeros(num_samps, dtype=np.complex64)

    f_ref = 50.0

    if random_phase:
        random_phase_deg = np.random.uniform(0, 360)
    else:
        random_phase_deg = 0.0

    phase_ref = (360.0 * f_ref * t + random_phase_deg) % 360.0

    prob_pd = (
        0.8 * gaussian_phase(phase_ref, 60, 20)
        + 1.0 * gaussian_phase(phase_ref, 200, 8)
    )

    prob_pd = prob_pd / np.max(prob_pd)

    rng = np.random.default_rng()

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

        pulse = A * np.exp(-tp / tau) * np.exp(
            1j * 2 * np.pi * f_offset * tp
        )

        end_idx = min(idx + pulse_len, num_samps)
        pd_signal[idx:end_idx] += pulse[:end_idx - idx]

    noise = (
        np.random.normal(0, 0.002, num_samps)
        + 1j * np.random.normal(0, 0.002, num_samps)
    ).astype(np.complex64)

    return pd_signal + noise


# =========================
# SPECTRE
# =========================

def compute_spectrum_dbm_per_bin(
        samples,
        rate,
        freq_center,
        nfft=4096,
        R=50,
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
        X = np.fft.fftshift(np.fft.fft(xw, n=nfft))

        vrms_bin = np.abs(X) / (nfft * coherent_gain * np.sqrt(2))
        p_w_bin = (vrms_bin ** 2) / R

        p_acc += p_w_bin

    p_mean = p_acc / n_blocks

    p_dbm_bin = 10 * np.log10(np.clip(p_mean / 1e-3, 1e-20, None))

    freqs = np.fft.fftshift(
        np.fft.fftfreq(nfft, d=1 / rate)
    ) + freq_center

    half_bw = rate / 2

    mask_edges = (
        (freqs >= freq_center - half_bw + edge_guard_hz) &
        (freqs <= freq_center + half_bw - edge_guard_hz)
    )

    return freqs[mask_edges], p_dbm_bin[mask_edges]


def plot_spectrum(freqs, psd_dbm, name="spectrum"):
    if len(freqs) == 0:
        print("Spectre vide.")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(freqs / 1e6, psd_dbm)
    plt.title("Spectre RX - dBm/bin approximatif")
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Puissance approximative (dBm/bin)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", bbox_inches="tight")
    plt.show()


# =========================
# TRAITEMENT PRPD
# =========================

def process_pd_signal_dbm(samples, rate, t_start=0.0, f_offset=2e6, R=50):
    samples = np.asarray(samples, dtype=np.complex64).ravel()
    N = len(samples)

    if N == 0:
        return np.array([]), np.array([])

    samples = samples - np.mean(samples)

    t = np.arange(N) / rate

    samples_bb = samples * np.exp(-1j * 2 * np.pi * f_offset * t)

    cutoff_if = 1e6
    sos_if = butter(4, cutoff_if / (rate / 2), btype="low", output="sos")
    samples_if = sosfiltfilt(sos_if, samples_bb)

    envelope_raw = np.abs(samples_if)

    cutoff_env = 10_000
    sos_env = butter(4, cutoff_env / (rate / 2), btype="low", output="sos")
    envelope = sosfiltfilt(sos_env, envelope_raw)

    noise_level = np.median(envelope)
    noise_std = np.std(envelope)

    threshold = (noise_level + 4.0 * noise_std) / 3

    min_distance = int(200e-6 * rate)

    peaks, props = find_peaks(
        envelope,
        height=threshold,
        distance=min_distance
    )

    if len(peaks) == 0:
        print("Aucun pic PRPD détecté.")
        return np.array([]), np.array([])

    t_peaks = t_start + peaks / rate

    cycle_time = t_peaks % 0.02
    phases = (cycle_time / 0.02) * 360.0

    amps = envelope[peaks]

    vrms = amps / np.sqrt(2)
    p_w = (vrms ** 2) / R

    amps_dbm = 10 * np.log10(np.clip(p_w / 1e-3, 1e-20, None))

    print(f"t_start = {t_start:.6f} s")
    print(f"Phase min = {np.min(phases):.2f} deg")
    print(f"Phase max = {np.max(phases):.2f} deg")

    return phases, amps_dbm


# =========================
# PLOT PRPD MULTIPLE
# =========================

def plot_prpd_multiple(acquisitions, name="PRPD_dBm"):
    if len(acquisitions) == 0:
        print("PRPD vide.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    all_phases = []
    all_amps = []

    for i, (phases, amps_dbm) in enumerate(acquisitions):
        if len(phases) == 0:
            continue

        ax1.scatter(
            phases,
            amps_dbm,
            s=15,
            alpha=0.7,
            label=f"Acquisition {i + 1}"
        )

        all_phases.extend(phases)
        all_amps.extend(amps_dbm)

    all_phases = np.array(all_phases)
    all_amps = np.array(all_amps)

    ax1.set_title(name)
    ax1.set_xlabel("Phase 50 Hz (degrés)")
    ax1.set_ylabel("Puissance impulsion approximative (dBm)")
    ax1.set_xlim(0, 360)
    ax1.grid(True)
    ax1.legend()

    if len(all_phases) > 0:
        h = ax2.hist2d(
            all_phases,
            all_amps,
            bins=[128, 50],
            cmap="jet"
        )
        fig.colorbar(h[3], ax=ax2, label="Densité")

    ax2.set_title("Heatmap PRPD globale")
    ax2.set_xlabel("Phase 50 Hz (degrés)")
    ax2.set_ylabel("Puissance impulsion approximative (dBm)")
    ax2.set_xlim(0, 360)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", bbox_inches="tight")
    plt.show()


# =========================
# TX/RX LOOPBACK
# =========================

def tx_rx_loopback_sync(usrp, tx_signal, freq, rate):
    max_amp = np.max(np.abs(tx_signal))

    if max_amp > 0:
        tx_signal = 0.5 * tx_signal / max_amp

    usrp.set_tx_rate(rate, CHANNEL)
    usrp.set_rx_rate(rate, CHANNEL)

    usrp.set_tx_freq(uhd.types.TuneRequest(freq), CHANNEL)
    usrp.set_rx_freq(uhd.types.TuneRequest(freq), CHANNEL)

    usrp.set_tx_gain(TX_GAIN, CHANNEL)
    usrp.set_rx_gain(RX_GAIN, CHANNEL)

    usrp.set_tx_antenna(TX_ANTENNA, CHANNEL)
    usrp.set_rx_antenna(RX_ANTENNA, CHANNEL)

    usrp.set_rx_dc_offset(True, CHANNEL)
    usrp.set_rx_iq_balance(True, CHANNEL)

    time.sleep(0.3)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [CHANNEL]

    rx_streamer = usrp.get_rx_stream(stream_args)
    tx_streamer = usrp.get_tx_stream(stream_args)

    num_samps = len(tx_signal)

    received = np.zeros(num_samps, dtype=np.complex64)

    rx_md = uhd.types.RXMetadata()
    tx_md = uhd.types.TXMetadata()

    current_time = usrp.get_time_now().get_real_secs()
    future_time = current_time + 0.1

    start_time = np.ceil(future_time / 0.02) * 0.02
    time_spec = uhd.types.TimeSpec(start_time)

    print(f"Temps SDR actuel = {current_time:.6f} s")
    print(f"TX/RX planifié à t = {start_time:.6f} s")

    rx_total = 0
    t_start = None

    def rx_worker():
        nonlocal rx_total, t_start

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        stream_cmd.num_samps = num_samps
        stream_cmd.stream_now = False
        stream_cmd.time_spec = time_spec

        rx_streamer.issue_stream_cmd(stream_cmd)

        buff = np.zeros((1, 4096), dtype=np.complex64)

        while rx_total < num_samps:
            n = rx_streamer.recv(buff, rx_md, timeout=5.0)

            if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
                print("RX error:", rx_md.strerror())
                continue

            if n > 0:
                if t_start is None:
                    t_start = rx_md.time_spec.get_real_secs()

                end = min(rx_total + n, num_samps)
                received[rx_total:end] = buff[0, :end - rx_total]
                rx_total = end

    rx_thread = threading.Thread(target=rx_worker)
    rx_thread.start()

    time.sleep(0.02)

    tx_md.start_of_burst = True
    tx_md.end_of_burst = False
    tx_md.has_time_spec = True
    tx_md.time_spec = time_spec

    chunk_size = 4096
    tx_total = 0

    for i in range(0, num_samps, chunk_size):
        chunk = tx_signal[i:i + chunk_size].astype(np.complex64)

        if i + chunk_size >= num_samps:
            tx_md.end_of_burst = True
        else:
            tx_md.end_of_burst = False

        sent = tx_streamer.send(chunk, tx_md)
        tx_total += sent

        tx_md.start_of_burst = False
        tx_md.has_time_spec = False

    rx_thread.join()

    print(f"TX envoyé : {tx_total}/{num_samps} samples")
    print(f"RX reçu   : {rx_total}/{num_samps} samples")

    if t_start is None:
        t_start = start_time

    return received[:rx_total], t_start


# =========================
# TEST PRPD
# =========================

def run_prpd_test(usrp, use_external_sync=False, test_name="test"):
    num_samps = int(DURATION * RATE)

    acquisitions = []

    for acq in range(N_ACQ):
        print(f"\n========== {test_name} | Acquisition {acq + 1}/{N_ACQ} ==========")

        if use_external_sync:
            sync_on_external_50hz_pps(usrp)

        # Comme vous l'avez très bien déduit, en loopback (TX et RX sur la même carte),
        # l'horloge est partagée. Le PRPD sera donc toujours parfaitement aligné 
        # si random_phase=False, que la synchro externe soit branchée ou non !
        # Le TX/RX ne permet PAS de valider la synchro externe.
        
        tx_signal = simulate_pd_signal(
            num_samps=num_samps,
            rate=RATE,
            f_offset=F_OFFSET,
            random_phase=False
        )

        rx_signal, t_start = tx_rx_loopback_sync(
            usrp=usrp,
            tx_signal=tx_signal,
            freq=FREQ,
            rate=RATE
        )

        if len(rx_signal) == 0:
            print("Aucun signal RX reçu.")
            continue

        # freqs, psd_dbm = compute_spectrum_dbm_per_bin(
        #     samples=rx_signal,
        #     rate=RATE,
        #     freq_center=FREQ,
        #     nfft=NFFT,
        #     R=R,
        #     remove_dc=True,
        #     edge_guard_hz=EDGE
        # )

        # plot_spectrum(
        #     freqs,
        #     psd_dbm,
        #     name=f"{test_name}_spectrum_acq_{acq + 1}"
        # )

        phases, amps_dbm = process_pd_signal_dbm(
            samples=rx_signal,
            rate=RATE,
            t_start=t_start,
            f_offset=F_OFFSET,
            R=R
        )

        print(f"Nombre de pulses détectés : {len(phases)}")

        acquisitions.append((phases, amps_dbm))

        time.sleep(0.3)

    plot_prpd_multiple(
        acquisitions,
        name=test_name
    )


# =========================
# TEST HARDWARE DE LA BROCHE PPS
# =========================

def test_pps_hardware(usrp, duration_s=2.0):
    print("\n==============================")
    print("VALIDATION MATERIELLE DE LA BROCHE PPS")
    print("==============================")
    print(f"Écoute des fronts montants sur la broche PPS pendant {duration_s} secondes...")
    
    usrp.set_time_source("external")
    time.sleep(0.1) # Laisser le temps à l'USRP de configurer la source
    
    pps_times = []
    start_time = time.time()
    
    last_pps = usrp.get_time_last_pps().get_real_secs()
    
    # On boucle et on regarde si le registre matériel de l'USRP se met à jour
    while time.time() - start_time < duration_s:
        curr_pps = usrp.get_time_last_pps().get_real_secs()
        if curr_pps != last_pps:
            pps_times.append(curr_pps)
            last_pps = curr_pps
            
    if len(pps_times) == 0:
        print("-> ECHEC : Aucun signal détecté sur la broche PPS ! Le 50Hz n'arrive pas au SDR.")
        return False
        
    deltas = np.diff(pps_times)
    
    # Filtrage des rebonds matériels (bounces) < 10 ms
    valid_deltas = deltas[deltas > 0.01]
    bounces = len(deltas) - len(valid_deltas)
    
    mean_delta = np.mean(valid_deltas) if len(valid_deltas) > 0 else 0
    std_delta = np.std(valid_deltas) if len(valid_deltas) > 0 else 0
    freq_est = 1.0 / mean_delta if mean_delta > 0 else 0
    
    print(f"-> SUCCES : {len(pps_times)} impulsions détectées par le SDR ! ({bounces} rebonds matériels ignorés)")
    print(f"-> Intervalle moyen (sans rebonds) : {mean_delta:.5f} secondes (Ecart-type: {std_delta:.5f} s)")
    print(f"-> Fréquence mesurée par le SDR : {freq_est:.2f} Hz")
    
    if len(deltas) > 0:
        print(f"-> Exemples des 10 premiers intervalles bruts : {np.round(deltas[:10], 5)}")
    
    if 49.0 <= freq_est <= 51.0 and std_delta < 0.001:
        print("-> PARFAIT : Le signal 50Hz est correctement reçu par la broche PPS de votre B200 !")
        return True
    else:
        print("-> ATTENTION : Le signal reçu n'est pas stable à 50 Hz.")
        return False


# =========================
# MAIN
# =========================

def main():
    print("Initialisation USRP...")

    usrp = uhd.usrp.MultiUSRP()
    
    # 1. Tester physiquement la broche PPS pour valider la synchro externe
    pps_ok = test_pps_hardware(usrp, duration_s=2.0)
    
    if not pps_ok:
        print("\nATTENTION : La synchro matérielle semble échouer. Les tests PRPD qui suivent ne seront pas alignés sur le vrai 50Hz.")
        time.sleep(2)

    # Initialisation une seule fois pour le test sans sync
    usrp.set_clock_source("internal")
    usrp.set_time_source("internal")
    usrp.set_time_now(uhd.types.TimeSpec(0.0))

    print("\n==============================")
    print("TEST 1 : SANS SYNC EXTERNE")
    print("==============================")

    run_prpd_test(
        usrp=usrp,
        use_external_sync=False,
        test_name="prpd_sans_sync_externe"
    )

    time.sleep(1.0)

    print("\n==============================")
    print("TEST 2 : AVEC SYNC EXTERNE 50 Hz PPS")
    print("==============================")

    run_prpd_test(
        usrp=usrp,
        use_external_sync=True,  # CORRECTION: Était False, doit être True pour activer le PPS
        test_name="prpd_avec_sync_externe_50hz_pps"
    )




if __name__ == "__main__":
    main()