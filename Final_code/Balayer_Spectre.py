import numpy as np
import matplotlib.pyplot as plt
import uhd
import time
import os

# =========================
# CONFIGURATION
# =========================

F_START = 100e6
F_STOP = 3.0e9   # Corrigé : 2.0 GHz (2.0e9) au lieu de 200 MHz (0.2e9)

RATE = 30e6      # 30 MHz est plus stable sur B200 en USB 3 (38 MHz peut créer des drops)
GAIN = 0        # Gain de 40 dB pour capter les signaux réels de l'antenne
CHANNEL = 0
ANTENNA = "RX2"

NFFT = 4096

DURATION_PER_STEP = 0.05 # 50 ms est amplement suffisant (donne environ 350 blocs FFT par palier)
SETTLE_TIME = 0.05       # Laisser 50 ms à l'oscillateur pour se stabiliser après chaque saut

OVERLAP = 0.20           # 20% de recouvrement pour combler les bords
STEP_FREQ = RATE * (1 - OVERLAP)

EDGE_MARGIN = 0.10       # On coupe 10% de chaque bord (atténués par les filtres analogiques)

OUTPUT_DIR = "./Main_figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# La fonction 'compute_psd_blocks' contenant des plt.show() dans une boucle a été supprimée
# pour éviter de faire planter l'ordinateur avec des milliers de fenêtres matplotlib.



def compute_psd_blocks_dbfs(samples, rate, freq_center, nfft=4096):
    samples = np.asarray(samples, dtype=np.complex64)
    samples = np.ravel(samples)

    n_blocks = len(samples) // nfft

    if n_blocks == 0:
        return np.array([]), np.array([])

    samples = samples[:n_blocks * nfft]
    blocks = samples.reshape(n_blocks, nfft)

    window = np.hanning(nfft)
    S_w = np.sum(window)

    psd_acc = np.zeros(nfft)

    for blk in blocks:

        # FFT
        X = np.fft.fftshift(
            np.fft.fft(blk * window, n=nfft)
        )

        # Normalisation par rapport au gain de la fenêtre
        # Un sinus complexe pleine échelle (amplitude 1.0) donnera 0 dBFS
        P_norm = (np.abs(X) / S_w) ** 2

        psd_acc += P_norm

    psd_mean = psd_acc / n_blocks

    # --- SUPPRESSION DU PIC CENTRAL (DC OFFSET / LO LEAKAGE) ---
    center_idx = nfft // 2
    if center_idx > 0 and center_idx < nfft - 1:
        psd_mean[center_idx] = (psd_mean[center_idx - 1] + psd_mean[center_idx + 1]) / 2.0

    # conversion dBFS
    psd_dbfs = 10 * np.log10(
        np.clip(psd_mean, 1e-20, None)
    )

    freqs = np.fft.fftshift(
        np.fft.fftfreq(nfft, d=1 / rate)
    )

    freqs = freqs + freq_center

    return freqs, psd_dbfs

def capture_fast(usrp, freq_center, rate, duration, gain):
    num_samps = int(rate * duration)

    usrp.set_rx_freq(uhd.types.TuneRequest(freq_center), CHANNEL)
    time.sleep(SETTLE_TIME)

    samples = usrp.recv_num_samps(
        num_samps,
        freq_center,
        rate,
        [CHANNEL],
        gain
    )

    samples = np.asarray(samples, dtype=np.complex64)
    samples = np.ravel(samples)

    return samples


def build_centers():
    first_center = F_START + RATE / 2
    last_center = F_STOP - RATE / 2

    centers = np.arange(first_center, last_center + STEP_FREQ, STEP_FREQ)

    centers = centers[(centers >= first_center) & (centers <= last_center)]

    return centers


def sweep_spectrum():
    print("Initialisation USRP...")

    usrp = uhd.usrp.MultiUSRP()

    usrp.set_rx_rate(RATE, CHANNEL)
    usrp.set_rx_gain(GAIN, CHANNEL)
    usrp.set_rx_antenna(ANTENNA, CHANNEL)
    usrp.set_rx_dc_offset(True, CHANNEL) # Activation de la correction DC matérielle

    centers = build_centers()

    all_freqs = []
    all_psd = []

    for i, fc in enumerate(centers):
        print(f"[{i + 1}/{len(centers)}] Capture autour de {fc / 1e6:.2f} MHz")

        try:
            samples = capture_fast(
                usrp,
                fc,
                RATE,
                DURATION_PER_STEP,
                GAIN
            )

            freqs, psd_db = compute_psd_blocks_dbfs(
                samples,
                RATE,
                fc,
                NFFT
            )

            if len(freqs) == 0:
                continue

            margin = int(EDGE_MARGIN * len(freqs))

            if margin > 0:
                freqs = freqs[margin:-margin]
                psd_db = psd_db[margin:-margin]

            mask = (freqs >= F_START) & (freqs <= F_STOP)

            freqs = freqs[mask]
            psd_db = psd_db[mask]

            all_freqs.append(freqs)
            all_psd.append(psd_db)

        except Exception as e:
            print(f"Erreur à {fc / 1e6:.2f} MHz : {e}")

    if len(all_freqs) == 0:
        raise RuntimeError("Aucune capture valide. Vérifie SDR, antenne, gain, RATE.")

    all_freqs = np.concatenate(all_freqs)
    all_psd = np.concatenate(all_psd)

    order = np.argsort(all_freqs)
    all_freqs = all_freqs[order]
    all_psd = all_psd[order]

    return all_freqs, all_psd


def plot_spectrum(freqs, psd_db):
    plt.figure(figsize=(18, 6))

    plt.plot(freqs / 1e6, psd_db, linewidth=0.6)

    plt.title(
        f"Sweep SDR : Spectre de {F_START / 1e6:.0f} MHz à {F_STOP / 1e6:.0f} MHz"
    )
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Puissance (dBFS)")
    plt.grid(True)

    plt.xlim(F_START / 1e6, F_STOP / 1e6)

    plt.tight_layout()

    filename = f"{OUTPUT_DIR}/sweep_{F_START/1e6:.0f}MHz_{F_STOP/1e6:.0f}MHz.png"
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"Image sauvegardée : {filename}")


def main():
    t0 = time.time()

    freqs, psd_db = sweep_spectrum()

    data_file = f"{OUTPUT_DIR}/sweep_{F_START/1e6:.0f}MHz_{F_STOP/1e6:.0f}MHz.npz"
    
    t1 = time.time()
    print("\nSweep terminé.")
    print(f"Temps total : {t1 - t0:.2f} s")
    
    np.savez(
        data_file,
        freqs=freqs,
        psd_db=psd_db
    )

    
    plot_spectrum(freqs, psd_db)

    


    print(f"Données sauvegardées : {data_file}")


if __name__ == "__main__":
    main()