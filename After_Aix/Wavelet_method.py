import numpy as np
import matplotlib.pyplot as plt
import uhd
import time
import json
import pywt

from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis


# =====================================================
# CONFIGURATION SDR
# =====================================================

F_START = 100e6
F_STOP  = 2e9

RATE = 12e6
GAIN = 76
CHANNEL = 0
ANTENNA = "RX2"

ACQ_DURATION = 0.08
STEP_HZ = 5e6

REMOVE_DC = True
LP_CUTOFF_HZ = 5e6


# =====================================================
# CONFIGURATION WAVELET
# =====================================================

WAVELET_NAME = "db4"
WAVELET_LEVEL = 4

WIN_SIZE = 4096
STEP_WIN = 2048

REF_JSON_PATH = "wavelet_reference_no_dp.json"


# =====================================================
# CHARGER RÉFÉRENCE JSON
# =====================================================

def load_reference_json(path):
    with open(path, "r") as f:
        refs = json.load(f)

    print("Référence chargée :", path)
    print("Nombre de bandes dans JSON :", len(refs["bands"]))

    return refs


def get_reference_for_freq(refs, freq):
    """
    Récupère la référence correspondant à la fréquence.
    Si la fréquence exacte n'existe pas, on prend la plus proche.
    """

    freq_keys = np.array([float(k) for k in refs["bands"].keys()])
    idx = np.argmin(np.abs(freq_keys - freq))

    closest_freq = freq_keys[idx]
    ref = refs["bands"][str(int(closest_freq))]

    return ref, closest_freq


# =====================================================
# ACQUISITION USRP
# =====================================================

def acquire_time_domain(usrp, freq_center, rate, duration, gain, antenna):
    num_samps = int(rate * duration)

    usrp.set_rx_rate(rate, CHANNEL)
    usrp.set_rx_freq(uhd.types.TuneRequest(freq_center), CHANNEL)
    usrp.set_rx_gain(gain, CHANNEL)
    usrp.set_rx_antenna(antenna, CHANNEL)

    usrp.set_rx_dc_offset(True, CHANNEL)
    usrp.set_rx_iq_balance(True, CHANNEL)

    time.sleep(0.05)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [CHANNEL]

    rx_streamer = usrp.get_rx_stream(stream_args)
    rx_md = uhd.types.RXMetadata()

    samples = np.zeros(num_samps, dtype=np.complex64)
    buff = np.zeros((1, 4096), dtype=np.complex64)

    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samps
    stream_cmd.stream_now = True

    rx_streamer.issue_stream_cmd(stream_cmd)

    total = 0

    while total < num_samps:
        n = rx_streamer.recv(buff, rx_md, timeout=3.0)

        if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
            print("RX error:", rx_md.strerror())
            continue

        if n > 0:
            end = min(total + n, num_samps)
            samples[total:end] = buff[0, :end-total]
            total = end

    return samples[:total]


# =====================================================
# PRETRAITEMENT
# =====================================================

def preprocess_samples(samples, rate):
    samples = np.asarray(samples, dtype=np.complex64).ravel()

    if len(samples) == 0:
        return np.array([])

    if REMOVE_DC:
        samples = samples - np.mean(samples)

    cutoff = min(LP_CUTOFF_HZ, 0.45 * rate)

    sos = butter(
        4,
        cutoff / (rate / 2),
        btype="low",
        output="sos"
    )

    samples_filt = sosfiltfilt(sos, samples)

    envelope = np.abs(samples_filt)

    return envelope


# =====================================================
# FEATURES WAVELET AVEC RÉFÉRENCE
# =====================================================

def compute_wavelet_features_with_reference(envelope, ref, rate):
    envelope = np.asarray(envelope, dtype=np.float64).ravel()

    if len(envelope) < WIN_SIZE:
        return None

    env_median = ref["envelope_median"]
    env_sigma = ref["envelope_sigma"] + 1e-20

    z = (envelope - env_median) / env_sigma

    times = []
    max_detail = []
    energy_detail = []
    kurt_detail = []

    for start in range(0, len(z) - WIN_SIZE, STEP_WIN):
        w = z[start:start + WIN_SIZE]

        coeffs = pywt.wavedec(
            w,
            WAVELET_NAME,
            level=WAVELET_LEVEL
        )

        D1 = coeffs[-1]
        D2 = coeffs[-2]
        D3 = coeffs[-3]

        details = np.concatenate([D1, D2, D3])

        times.append((start + WIN_SIZE / 2) / rate)
        max_detail.append(np.max(np.abs(details)))
        energy_detail.append(np.sum(details ** 2))
        kurt_detail.append(kurtosis(details, fisher=True))

    return {
        "times": np.array(times),
        "max_detail": np.array(max_detail),
        "energy_detail": np.array(energy_detail),
        "kurt_detail": np.array(kurt_detail),
    }


# =====================================================
# SCORE DP POUR UNE BANDE AVEC JSON
# =====================================================

def wavelet_dp_score_with_json(samples, rate, ref):
    envelope = preprocess_samples(samples, rate)

    features = compute_wavelet_features_with_reference(
        envelope=envelope,
        ref=ref,
        rate=rate
    )

    if features is None:
        return {
            "score": np.nan,
            "n_suspect": 0,
            "n_windows": 0,
            "suspect_rate": np.nan,
            "max_peak": np.nan,
            "kurt_peak": np.nan,
            "energy_peak_db": np.nan,
        }

    max_detail = features["max_detail"]
    energy_detail = features["energy_detail"]
    kurt_detail = features["kurt_detail"]

    T_max = ref["max_detail_threshold"]
    T_energy = ref["energy_detail_threshold"]
    T_kurt = ref["kurt_detail_threshold"]

    suspect_max = max_detail > T_max
    suspect_energy = energy_detail > T_energy
    suspect_kurt = kurt_detail > T_kurt

    suspect = suspect_max | suspect_energy | suspect_kurt

    n_windows = len(max_detail)
    n_suspect = int(np.sum(suspect))
    suspect_rate = n_suspect / max(n_windows, 1)

    max_peak = np.max(max_detail)
    kurt_peak = np.max(kurt_detail)
    energy_peak_db = 10 * np.log10(np.max(energy_detail) + 1e-20)

    # Score basé sur référence JSON
    score = 0

    if suspect_rate > 0.005:
        score += 20
    if suspect_rate > 0.01:
        score += 20
    if suspect_rate > 0.03:
        score += 20

    if max_peak > T_max:
        score += 15

    if kurt_peak > T_kurt:
        score += 15

    if np.max(energy_detail) > T_energy:
        score += 10

    score = min(score, 100)

    return {
        "score": score,
        "n_suspect": n_suspect,
        "n_windows": n_windows,
        "suspect_rate": suspect_rate,

        "n_suspect_max": int(np.sum(suspect_max)),
        "n_suspect_energy": int(np.sum(suspect_energy)),
        "n_suspect_kurt": int(np.sum(suspect_kurt)),

        "max_peak": max_peak,
        "kurt_peak": kurt_peak,
        "energy_peak_db": energy_peak_db,

        "T_max": T_max,
        "T_kurt": T_kurt,
        "T_energy_db": 10 * np.log10(T_energy + 1e-20),
    }


# =====================================================
# SCAN FREQUENTIEL
# =====================================================

def scan_pd_wavelet_with_json():
    refs = load_reference_json(REF_JSON_PATH)

    print("Initialisation USRP...")
    usrp = uhd.usrp.MultiUSRP()

    freqs = np.arange(F_START, F_STOP + STEP_HZ, STEP_HZ)

    scores = []
    suspect_rates = []
    n_suspects = []
    n_suspects_max = []
    n_suspects_energy = []
    n_suspects_kurt = []

    max_peaks = []
    kurt_peaks = []
    energy_peaks_db = []

    ref_freqs = []

    for i, freq in enumerate(freqs):
        ref, ref_freq = get_reference_for_freq(refs, freq)

        print(f"\n[{i+1}/{len(freqs)}] Acquisition à {freq/1e6:.1f} MHz")
        print(f"  Référence utilisée : {ref_freq/1e6:.1f} MHz")

        samples = acquire_time_domain(
            usrp=usrp,
            freq_center=freq,
            rate=RATE,
            duration=ACQ_DURATION,
            gain=GAIN,
            antenna=ANTENNA
        )

        result = wavelet_dp_score_with_json(
            samples=samples,
            rate=RATE,
            ref=ref
        )

        scores.append(result["score"])
        suspect_rates.append(result["suspect_rate"])
        n_suspects.append(result["n_suspect"])

        n_suspects_max.append(result["n_suspect_max"])
        n_suspects_energy.append(result["n_suspect_energy"])
        n_suspects_kurt.append(result["n_suspect_kurt"])

        max_peaks.append(result["max_peak"])
        kurt_peaks.append(result["kurt_peak"])
        energy_peaks_db.append(result["energy_peak_db"])

        ref_freqs.append(ref_freq)

        print(
            f"  Score DP = {result['score']:.1f}% | "
            f"Fenêtres suspectes = {result['n_suspect']}/{result['n_windows']} | "
            f"Taux = {100*result['suspect_rate']:.2f}%"
        )

        print(
            f"  Suspect Max={result['n_suspect_max']} | "
            f"Energy={result['n_suspect_energy']} | "
            f"Kurt={result['n_suspect_kurt']}"
        )

        print(
            f"  MaxCoef={result['max_peak']:.2f} / T={result['T_max']:.2f} | "
            f"KurtMax={result['kurt_peak']:.2f} / T={result['T_kurt']:.2f} | "
            f"EnergyMax={result['energy_peak_db']:.2f} dB / T={result['T_energy_db']:.2f} dB"
        )

    return {
        "freqs": freqs,
        "ref_freqs": np.array(ref_freqs),
        "scores": np.array(scores),
        "suspect_rates": np.array(suspect_rates),
        "n_suspects": np.array(n_suspects),
        "n_suspects_max": np.array(n_suspects_max),
        "n_suspects_energy": np.array(n_suspects_energy),
        "n_suspects_kurt": np.array(n_suspects_kurt),
        "max_peaks": np.array(max_peaks),
        "kurt_peaks": np.array(kurt_peaks),
        "energy_peaks_db": np.array(energy_peaks_db),
    }


# =====================================================
# VISUALISATION
# =====================================================

def plot_wavelet_scan_with_json(results):
    freqs_mhz = results["freqs"] / 1e6

    plt.figure(figsize=(14, 5))
    plt.plot(freqs_mhz, results["scores"], marker="o")
    plt.title("Score DP par fréquence - Wavelet avec référence JSON sans DP")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Score DP (%)")
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(freqs_mhz, 100 * results["suspect_rates"], marker="o")
    plt.title("Taux de fenêtres suspectes par fréquence")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Fenêtres suspectes (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(freqs_mhz, results["n_suspects_max"], label="Max coef")
    plt.plot(freqs_mhz, results["n_suspects_energy"], label="Énergie")
    plt.plot(freqs_mhz, results["n_suspects_kurt"], label="Kurtosis")
    plt.title("Nombre de fenêtres suspectes par critère")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Nombre de fenêtres")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(freqs_mhz, results["max_peaks"], marker="o")
    plt.title("Maximum des coefficients Wavelet")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Max coefficient normalisé")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(freqs_mhz, results["kurt_peaks"], marker="o")
    plt.title("Kurtosis Wavelet maximale")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Kurtosis max")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(freqs_mhz, results["energy_peaks_db"], marker="o")
    plt.title("Énergie Wavelet locale maximale")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Énergie locale max (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best_idx = np.nanargmax(results["scores"])

    print("\n===== Résumé scan DP avec référence JSON =====")
    print(f"Fréquence la plus suspecte : {freqs_mhz[best_idx]:.1f} MHz")
    print(f"Score DP max              : {results['scores'][best_idx]:.1f}%")
    print(f"Taux fenêtres suspectes   : {100*results['suspect_rates'][best_idx]:.2f}%")
    print(f"Fenêtres suspectes        : {results['n_suspects'][best_idx]}")
    print(f"Max coefficient wavelet   : {results['max_peaks'][best_idx]:.2f}")
    print(f"Kurtosis max              : {results['kurt_peaks'][best_idx]:.2f}")
    print(f"Énergie max               : {results['energy_peaks_db'][best_idx]:.2f} dB")


# =====================================================
# MAIN
# =====================================================

def main():
    results = scan_pd_wavelet_with_json()
    plot_wavelet_scan_with_json(results)
    return results


results = main()