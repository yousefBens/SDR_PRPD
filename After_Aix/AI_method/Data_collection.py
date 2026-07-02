import os
import json
import time
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import uhd
import pywt

from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import kurtosis, skew




F_START = 100e6
F_STOP  = 2e9

RATE = 12e6
GAIN = 76
CHANNEL = 0
ANTENNA = "RX2"

ACQ_DURATION = 0.1
STEP_HZ = 5e6

REMOVE_DC = True
LP_CUTOFF_HZ = 5e6




N_REPEATS_NO_DP = 20
N_REPEATS_WITH_DP = 20

OUTPUT_DIR = "/home/yousef/Documents/testing_scripts/GEVernova/After_Aix/AI_method/Dataset"

SAVE_RAW_SAMPLES = False   # True si tu veux aussi sauvegarder les signaux IQ .npz
RAW_MAX_SAMPLES = 300000   # limitation si SAVE_RAW_SAMPLES=True




WAVELET_NAME = "db4"
WAVELET_LEVEL = 4

WIN_SIZE = 4096
STEP_WIN = 2048

PEAK_DISTANCE_US = 15
PEAK_K = 8



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




def robust_median_sigma(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad + 1e-20
    return med, sigma


def safe_db20(x):
    return 20 * np.log10(np.maximum(x, 1e-20))


def safe_db10(x):
    return 10 * np.log10(np.maximum(x, 1e-20))




def preprocess_samples(samples, rate):
    samples = np.asarray(samples, dtype=np.complex64).ravel()

    if len(samples) == 0:
        return np.array([]), np.array([])

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

    return samples_filt, envelope




def extract_time_features(envelope, rate):
    if len(envelope) == 0:
        return {}

    env = np.asarray(envelope, dtype=np.float64)

    med, sig = robust_median_sigma(env)
    z = (env - med) / sig

    threshold = med + PEAK_K * sig
    min_distance = int(PEAK_DISTANCE_US * 1e-6 * rate)

    peaks, props = find_peaks(
        env,
        height=threshold,
        distance=min_distance
    )

    peak_vals = env[peaks] if len(peaks) > 0 else np.array([0.0])

    rms = np.sqrt(np.mean(env ** 2))
    crest = np.max(env) / (rms + 1e-20)

    features = {
        "env_mean": float(np.mean(env)),
        "env_std": float(np.std(env)),
        "env_median": float(med),
        "env_sigma_robust": float(sig),
        "env_rms": float(rms),

        "env_max": float(np.max(env)),
        "env_p95": float(np.percentile(env, 95)),
        "env_p99": float(np.percentile(env, 99)),
        "env_p999": float(np.percentile(env, 99.9)),

        "env_max_db": float(safe_db20(np.max(env))),
        "env_p999_db": float(safe_db20(np.percentile(env, 99.9))),

        "crest_factor": float(crest),
        "crest_factor_db": float(safe_db20(crest)),

        "env_kurtosis": float(kurtosis(env, fisher=True)),
        "env_skewness": float(skew(env)),

        "z_max": float(np.max(z)),
        "z_p999": float(np.percentile(z, 99.9)),

        "peak_threshold": float(threshold),
        "n_peaks": int(len(peaks)),
        "peak_rate_per_s": float(len(peaks) / (len(env) / rate)),
        "peak_max": float(np.max(peak_vals)),
        "peak_mean": float(np.mean(peak_vals)),
        "peak_energy": float(np.sum(peak_vals ** 2)),
        "peak_energy_db": float(safe_db10(np.sum(peak_vals ** 2))),
    }

    return features



def extract_wavelet_features(envelope, rate):
    env = np.asarray(envelope, dtype=np.float64)

    if len(env) < WIN_SIZE:
        return {}

    med, sig = robust_median_sigma(env)
    z = (env - med) / sig

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

        max_detail.append(np.max(np.abs(details)))
        energy_detail.append(np.sum(details ** 2))
        kurt_detail.append(kurtosis(details, fisher=True))

    max_detail = np.array(max_detail)
    energy_detail = np.array(energy_detail)
    kurt_detail = np.array(kurt_detail)

    features = {
        "wav_n_windows": int(len(max_detail)),

        "wav_max_mean": float(np.mean(max_detail)),
        "wav_max_std": float(np.std(max_detail)),
        "wav_max_median": float(np.median(max_detail)),
        "wav_max_p95": float(np.percentile(max_detail, 95)),
        "wav_max_p99": float(np.percentile(max_detail, 99)),
        "wav_max_p999": float(np.percentile(max_detail, 99.9)),
        "wav_max_global": float(np.max(max_detail)),

        "wav_energy_mean": float(np.mean(energy_detail)),
        "wav_energy_std": float(np.std(energy_detail)),
        "wav_energy_median": float(np.median(energy_detail)),
        "wav_energy_p95": float(np.percentile(energy_detail, 95)),
        "wav_energy_p99": float(np.percentile(energy_detail, 99)),
        "wav_energy_p999": float(np.percentile(energy_detail, 99.9)),
        "wav_energy_global": float(np.max(energy_detail)),
        "wav_energy_global_db": float(safe_db10(np.max(energy_detail))),

        "wav_kurt_mean": float(np.mean(kurt_detail)),
        "wav_kurt_std": float(np.std(kurt_detail)),
        "wav_kurt_median": float(np.median(kurt_detail)),
        "wav_kurt_p95": float(np.percentile(kurt_detail, 95)),
        "wav_kurt_p99": float(np.percentile(kurt_detail, 99)),
        "wav_kurt_p999": float(np.percentile(kurt_detail, 99.9)),
        "wav_kurt_global": float(np.max(kurt_detail)),
    }

    return features




def extract_spectral_features(samples_filt, rate):
    x = np.asarray(samples_filt, dtype=np.complex64).ravel()

    if len(x) == 0:
        return {}

    nfft = 4096

    if len(x) < nfft:
        return {}

    x = x[:len(x) // nfft * nfft]
    blocks = x.reshape(-1, nfft)

    window = np.hanning(nfft)
    X = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)

    power = np.mean(np.abs(X) ** 2, axis=0)

    power_db = safe_db10(power)

    features = {
        "spec_power_mean_db": float(np.mean(power_db)),
        "spec_power_std_db": float(np.std(power_db)),
        "spec_power_max_db": float(np.max(power_db)),
        "spec_power_p95_db": float(np.percentile(power_db, 95)),
        "spec_power_p99_db": float(np.percentile(power_db, 99)),
        "spec_flatness": float(
            np.exp(np.mean(np.log(power + 1e-20))) / (np.mean(power) + 1e-20)
        ),
    }

    return features




def extract_all_features(samples, freq_center, label, repeat_idx, rate):
    samples_filt, envelope = preprocess_samples(samples, rate)

    features = {
        "timestamp": datetime.now().isoformat(),
        "label": int(label),  # 0 = sans DP, 1 = avec DP
        "label_name": "with_dp" if label == 1 else "no_dp",
        "repeat_idx": int(repeat_idx),
        "freq_center_hz": float(freq_center),
        "freq_center_mhz": float(freq_center / 1e6),
        "rate_hz": float(rate),
        "gain": float(GAIN),
        "acq_duration_s": float(ACQ_DURATION),
        "n_samples": int(len(samples)),
    }

    features.update(extract_time_features(envelope, rate))
    features.update(extract_wavelet_features(envelope, rate))
    features.update(extract_spectral_features(samples_filt, rate))

    return features




def prepare_output_folder():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "raw"), exist_ok=True)

    meta = {
        "created_at": datetime.now().isoformat(),
        "description": "Dataset features for Partial Discharge detection using SDR",
        "f_start_hz": F_START,
        "f_stop_hz": F_STOP,
        "step_hz": STEP_HZ,
        "rate_hz": RATE,
        "gain": GAIN,
        "channel": CHANNEL,
        "antenna": ANTENNA,
        "acq_duration_s": ACQ_DURATION,
        "remove_dc": REMOVE_DC,
        "lp_cutoff_hz": LP_CUTOFF_HZ,
        "wavelet": WAVELET_NAME,
        "wavelet_level": WAVELET_LEVEL,
        "win_size": WIN_SIZE,
        "step_win": STEP_WIN,
        "n_repeats_no_dp": N_REPEATS_NO_DP,
        "n_repeats_with_dp": N_REPEATS_WITH_DP,
    }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)

    return meta


def save_features_csv(features_list, csv_path):
    if len(features_list) == 0:
        return

    keys = sorted(set().union(*(d.keys() for d in features_list)))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in features_list:
            writer.writerow(row)


def save_features_json(features_list, json_path):
    with open(json_path, "w") as f:
        json.dump(features_list, f, indent=4)


def save_raw_if_needed(samples, label_name, freq, repeat_idx):
    if not SAVE_RAW_SAMPLES:
        return

    n = min(len(samples), RAW_MAX_SAMPLES)
    filename = f"{label_name}_rep{repeat_idx}_freq_{int(freq)}.npz"
    path = os.path.join(OUTPUT_DIR, "raw", filename)

    np.savez_compressed(
        path,
        samples=samples[:n],
        freq_center_hz=freq,
        rate_hz=RATE,
        label_name=label_name
    )




def collect_dataset_phase(usrp, label, n_repeats):
    label_name = "with_dp" if label == 1 else "no_dp"
    freqs = np.arange(F_START, F_STOP + STEP_HZ, STEP_HZ)

    all_features = []

    print("\n=====================================================")
    print(f"Début collecte : {label_name}")
    print("=====================================================")

    for rep in range(n_repeats):
        print(f"\n------ Répétition {rep+1}/{n_repeats} : {label_name} ------")

        for i, freq in enumerate(freqs):
            print(f"[{i+1}/{len(freqs)}] {label_name} | rep={rep+1} | freq={freq/1e6:.1f} MHz")

            samples = acquire_time_domain(
                usrp=usrp,
                freq_center=freq,
                rate=RATE,
                duration=ACQ_DURATION,
                gain=GAIN,
                antenna=ANTENNA
            )
            # print(samples.shape)
            # plt.figure()
            # plt.plot(samples)
            # plt.show()

            features = extract_all_features(
                samples=samples,
                freq_center=freq,
                label=label,
                repeat_idx=rep,
                rate=RATE
            )

            all_features.append(features)
            save_raw_if_needed(samples, label_name, freq, rep)

            print(
                f"  Peaks={features.get('n_peaks', np.nan)} | "
                f"WavMaxP999={features.get('wav_max_p999', np.nan):.2f} | "
                f"WavKurtP999={features.get('wav_kurt_p999', np.nan):.2f} | "
                f"Crest={features.get('crest_factor_db', np.nan):.2f} dB"
            )

    return all_features




def quick_plot(features_list):
    if len(features_list) == 0:
        return

    labels = np.array([d["label"] for d in features_list])
    freqs = np.array([d["freq_center_mhz"] for d in features_list])
    wav = np.array([d.get("wav_kurt_p999", np.nan) for d in features_list])
    peaks = np.array([d.get("peak_rate_per_s", np.nan) for d in features_list])

    plt.figure(figsize=(14, 5))
    plt.scatter(freqs[labels == 0], wav[labels == 0], label="Sans DP", alpha=0.7)
    plt.scatter(freqs[labels == 1], wav[labels == 1], label="Avec DP", alpha=0.7)
    plt.title("Feature wavelet kurtosis p99.9 par fréquence")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("wav_kurt_p999")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.scatter(freqs[labels == 0], peaks[labels == 0], label="Sans DP", alpha=0.7)
    plt.scatter(freqs[labels == 1], peaks[labels == 1], label="Avec DP", alpha=0.7)
    plt.title("Taux de pics par fréquence")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("peak_rate_per_s")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()




def main():
    prepare_output_folder()

    print("Initialisation USRP...")
    usrp = uhd.usrp.MultiUSRP()

    # Phase 1 : sans DP
    features_no_dp = collect_dataset_phase(
        usrp=usrp,
        label=0,
        n_repeats=N_REPEATS_NO_DP
    )

    print("\n=====================================================")
    print("Collecte SANS DP terminée.")
    print("Maintenant applique les DP physiquement.")
    print("Quand les DP sont présentes/stables, appuie sur Entrée.")
    print("=====================================================")

    input("Appuie sur Entrée pour commencer la collecte AVEC DP...")

    # Phase 2 : avec DP
    features_with_dp = collect_dataset_phase(
        usrp=usrp,
        label=1,
        n_repeats=N_REPEATS_WITH_DP
    )

    # Fusion dataset
    all_features = features_no_dp + features_with_dp

    csv_path = os.path.join(OUTPUT_DIR, "features_dataset.csv")
    json_path = os.path.join(OUTPUT_DIR, "features_dataset.json")

    save_features_csv(all_features, csv_path)
    save_features_json(all_features, json_path)

    print("\n=====================================================")
    print("Dataset terminé.")
    print("Dossier :", OUTPUT_DIR)
    print("CSV    :", csv_path)
    print("JSON   :", json_path)
    print("=====================================================")

    quick_plot(all_features)

    return all_features

if __name__ == "__main__":
    main()