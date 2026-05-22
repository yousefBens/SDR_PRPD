import numpy as np
import matplotlib.pyplot as plt
import uhd
import time
import os

# =========================
# CONFIGURATION
# =========================

F_START = 100e6
F_STOP = 0.2e9

RATE = 38e6
GAIN = 0
CHANNEL = 0
ANTENNA = "RX2"

NFFT = 4096

DURATION_PER_STEP = 0.2
SETTLE_TIME = 0.02

OVERLAP = 0.3
STEP_FREQ = RATE * (1 - OVERLAP)

EDGE_MARGIN = 0.03

OUTPUT_DIR = "./Main_figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_psd_blocks(samples, rate, freq_center, nfft=4096):
    samples = np.asarray(samples, dtype=np.complex64)
    samples = np.ravel(samples)

    n_blocks = len(samples) // nfft

    if n_blocks == 0:
        return np.array([]), np.array([])

    samples = samples[:n_blocks * nfft]
    blocks = samples.reshape(n_blocks, nfft)

    window = np.hanning(nfft)

    psd_acc = np.zeros(nfft)

    for blk in blocks:
        X = np.fft.fftshift(np.fft.fft(blk * window, n=nfft))
        plt.figure()
        plt.plot(X)
        plt.show()
        P = (np.abs(X) ** 2) / (nfft)
        psd_acc += P

    psd_mean = psd_acc / n_blocks
    psd_db = 10 * np.log10(psd_mean + 1e-20)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / rate))
    freqs = freqs + freq_center

    return freqs, psd_db



def compute_psd_blocks_dbm(samples, rate, freq_center, nfft=4096, R=50):
    samples = np.asarray(samples, dtype=np.complex64)
    samples = np.ravel(samples)

    n_blocks = len(samples) // nfft

    if n_blocks == 0:
        return np.array([]), np.array([])

    samples = samples[:n_blocks * nfft]
    blocks = samples.reshape(n_blocks, nfft)

    window = np.hanning(nfft)

    psd_acc = np.zeros(nfft)

    for blk in blocks:

        # FFT
        X = np.fft.fftshift(
            np.fft.fft(blk * window, n=nfft)
        )

        
        # amplitude RMS par bin
        Vrms = np.abs(X) / (nfft * np.sqrt(2))

        # puissance électrique
        P_w = (Vrms ** 2) / R

        psd_acc += P_w
        
        # plt.figure()
        # plt.plot(psd_acc)
        # plt.show()

    psd_mean = psd_acc / n_blocks

    # conversion dBm
    psd_dbm = 10 * np.log10(
        np.clip(psd_mean / 1e-3, 1e-20, None)
    )

    freqs = np.fft.fftshift(
        np.fft.fftfreq(nfft, d=1 / rate)
    )

    freqs = freqs + freq_center

    return freqs, psd_dbm

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

            freqs, psd_db = compute_psd_blocks_dbm(
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
    plt.ylabel("Puissance relative (dB)")
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