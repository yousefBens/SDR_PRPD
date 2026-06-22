import numpy as np
import matplotlib.pyplot as plt
import uhd
import time
import json
import pywt

from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis
from scipy.ndimage import median_filter, uniform_filter1d


# =====================================================
# CONFIGURATION SDR
# =====================================================

F_START = 100e6
F_STOP  = 2e9

RATE = 12e6
GAIN = 40
CHANNEL = 0
ANTENNA = "RX2"

ACQ_DURATION = 0.08
STEP_HZ = 5e6

REMOVE_DC = True
LP_CUTOFF_HZ = 5e6

REF_JSON_PATH = "wavelet_reference_no_dp.json"


# =====================================================
# CONFIGURATION WAVELET
# =====================================================

WAVELET_NAME = "db4"
WAVELET_LEVEL = 4

WIN_SIZE = 4096
STEP_WIN = 2048


# =====================================================
# CHARGER RÉFÉRENCE JSON
# =====================================================

def load_reference_json(path):
    with open(path, "r") as f:
        refs = json.load(f)

    print("Référence chargée :", path)
    print("Nombre de bandes :", len(refs["bands"]))

    return refs


def get_reference_for_freq(refs, freq):
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
# PRÉTRAITEMENT
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
# WAVELET FEATURES AVEC JSON
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
# SCORE PAR BANDE
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
            "n_suspect_max": 0,
            "n_suspect_energy": 0,
            "n_suspect_kurt": 0,
            "max_peak": np.nan,
            "kurt_peak": np.nan,
            "energy_peak_db": np.nan,
            "max_ratio": np.nan,
            "energy_ratio": np.nan,
            "kurt_ratio": np.nan,
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

    # Fenêtre suspecte seulement si au moins 2 critères sont actifs
    suspect_count = (
        suspect_max.astype(int)
        + suspect_energy.astype(int)
        + suspect_kurt.astype(int)
    )

    suspect = suspect_count >= 2

    n_windows = len(max_detail)
    n_suspect = int(np.sum(suspect))
    suspect_rate = n_suspect / max(n_windows, 1)

    max_peak = np.max(max_detail)
    kurt_peak = np.max(kurt_detail)
    energy_peak = np.max(energy_detail)

    max_ratio = np.percentile(max_detail, 99.5) / (ref["max_detail_p999"] + 1e-20)
    energy_ratio = np.percentile(energy_detail, 99.5) / (ref["energy_detail_p999"] + 1e-20)
    kurt_ratio = np.percentile(kurt_detail, 99.5) / (ref["kurt_detail_p999"] + 1e-20)

    score = 0

    if suspect_rate > 0.01:
        score += 25
    if suspect_rate > 0.03:
        score += 25

    if max_ratio > 1.5:
        score += 15
    if energy_ratio > 1.5:
        score += 15
    if kurt_ratio > 1.5:
        score += 20

    # Pénalité si seulement un ou deux événements isolés
    if n_suspect <= 2:
        score = min(score, 30)

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
        "energy_peak_db": 10 * np.log10(energy_peak + 1e-20),

        "max_ratio": max_ratio,
        "energy_ratio": energy_ratio,
        "kurt_ratio": kurt_ratio,
    }


# =====================================================
# SCAN FRÉQUENTIEL
# =====================================================

def scan_pd_wavelet_with_json():
    refs = load_reference_json(REF_JSON_PATH)

    print("Initialisation USRP...")
    usrp = uhd.usrp.MultiUSRP()

    freqs = np.arange(F_START, F_STOP + STEP_HZ, STEP_HZ)

    results = {
        "freqs": [],
        "scores": [],
        "suspect_rates": [],
        "n_suspects": [],
        "n_suspects_max": [],
        "n_suspects_energy": [],
        "n_suspects_kurt": [],
        "max_peaks": [],
        "kurt_peaks": [],
        "energy_peaks_db": [],
        "max_ratios": [],
        "energy_ratios": [],
        "kurt_ratios": [],
    }

    for i, freq in enumerate(freqs):
        ref, ref_freq = get_reference_for_freq(refs, freq)

        print(f"\n[{i+1}/{len(freqs)}] Acquisition à {freq/1e6:.1f} MHz")
        print(f"Référence utilisée : {ref_freq/1e6:.1f} MHz")

        samples = acquire_time_domain(
            usrp=usrp,
            freq_center=freq,
            rate=RATE,
            duration=ACQ_DURATION,
            gain=GAIN,
            antenna=ANTENNA
        )

        result = wavelet_dp_score_with_json(samples, RATE, ref)

        results["freqs"].append(freq)
        results["scores"].append(result["score"])
        results["suspect_rates"].append(result["suspect_rate"])
        results["n_suspects"].append(result["n_suspect"])
        results["n_suspects_max"].append(result["n_suspect_max"])
        results["n_suspects_energy"].append(result["n_suspect_energy"])
        results["n_suspects_kurt"].append(result["n_suspect_kurt"])
        results["max_peaks"].append(result["max_peak"])
        results["kurt_peaks"].append(result["kurt_peak"])
        results["energy_peaks_db"].append(result["energy_peak_db"])
        results["max_ratios"].append(result["max_ratio"])
        results["energy_ratios"].append(result["energy_ratio"])
        results["kurt_ratios"].append(result["kurt_ratio"])

        print(
            f"Score brut = {result['score']:.1f}% | "
            f"Fenêtres suspectes = {result['n_suspect']}/{result['n_windows']} | "
            f"Taux = {100*result['suspect_rate']:.2f}%"
        )

        print(
            f"Ratios : Max={result['max_ratio']:.2f} | "
            f"Energy={result['energy_ratio']:.2f} | "
            f"Kurt={result['kurt_ratio']:.2f}"
        )

    for key in results:
        results[key] = np.array(results[key])

    return results


# =====================================================
# POST-TRAITEMENT COHÉRENCE FRÉQUENTIELLE
# =====================================================

def postprocess_dp_score(raw_scores, median_size=5, smooth_size=9):
    raw_scores = np.asarray(raw_scores, dtype=float)

    score_med = median_filter(raw_scores, size=median_size, mode="nearest")
    score_smooth = uniform_filter1d(score_med, size=smooth_size, mode="nearest")

    isolated = (raw_scores > 60) & (score_smooth < 30)

    final_score = raw_scores.copy()
    final_score[isolated] *= 0.25

    final_score = 0.4 * final_score + 0.6 * score_smooth
    final_score = np.clip(final_score, 0, 100)

    return final_score, score_med, score_smooth, isolated


# =====================================================
# VISUALISATION FINALE
# =====================================================

def plot_final_results(results):
    freqs_mhz = results["freqs"] / 1e6
    raw_scores = results["scores"]

    final_score, score_med, score_smooth, isolated = postprocess_dp_score(
        raw_scores,
        median_size=5,
        smooth_size=9
    )

    plt.figure(figsize=(14, 5))
    plt.plot(freqs_mhz, raw_scores, alpha=0.35, label="Score brut")
    plt.plot(freqs_mhz, score_smooth, linewidth=2, label="Tendance large bande")
    plt.plot(freqs_mhz, final_score, linewidth=2, label="Score final cohérent")
    plt.scatter(
        freqs_mhz[isolated],
        raw_scores[isolated],
        marker="x",
        s=80,
        label="Pics isolés pénalisés"
    )

    plt.title("Score DP corrigé par cohérence fréquentielle")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Score DP (%)")
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.legend()
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
    plt.plot(freqs_mhz, results["n_suspects_max"], label="Critère max coef")
    plt.plot(freqs_mhz, results["n_suspects_energy"], label="Critère énergie")
    plt.plot(freqs_mhz, results["n_suspects_kurt"], label="Critère kurtosis")
    plt.title("Nombre de fenêtres suspectes par critère")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Nombre de fenêtres")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(freqs_mhz, results["max_ratios"], label="Ratio max")
    plt.plot(freqs_mhz, results["energy_ratios"], label="Ratio énergie")
    plt.plot(freqs_mhz, results["kurt_ratios"], label="Ratio kurtosis")
    plt.axhline(1.5, linestyle="--", label="Seuil ratio 1.5")
    plt.title("Ratios par rapport à la référence JSON sans DP")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Ratio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    best_idx = np.nanargmax(final_score)

    print("\n===== Résumé final =====")
    print(f"Fréquence la plus suspecte : {freqs_mhz[best_idx]:.1f} MHz")
    print(f"Score brut max            : {np.nanmax(raw_scores):.1f}%")
    print(f"Score final max           : {np.nanmax(final_score):.1f}%")
    print(f"Score final moyen         : {np.nanmean(final_score):.1f}%")
    print(f"Nombre pics isolés        : {np.sum(isolated)}")

    if np.nanmean(final_score) > 50:
        print("Conclusion : signature large bande compatible avec DP.")
    elif np.nanmean(final_score) > 25:
        print("Conclusion : signature DP possible, à confirmer.")
    else:
        print("Conclusion : pas de signature large bande forte.")

    return final_score


# =====================================================
# MAIN
# =====================================================

def main():
    results = scan_pd_wavelet_with_json()
    final_score = plot_final_results(results)
    return results, final_score


results, final_score = main()