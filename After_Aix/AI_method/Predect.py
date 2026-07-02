import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uhd
import pywt
import joblib

from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import kurtosis, skew




F_START = 100e6
F_STOP = 2e9
STEP_HZ = 5e6

RATE = 12e6
GAIN = 40

CHANNEL = 0
ANTENNA = "RX2"

ACQ_DURATION = 0.1

REMOVE_DC = True
LP_CUTOFF_HZ = 5e6




PEAK_K = 8
PEAK_DISTANCE_US = 20

WAVELET_NAME = "db4"
WAVELET_LEVEL = 4

WIN_SIZE = 4096
STEP_WIN = 2048




MODEL_DIR = "/home/yousef/Documents/testing_scripts/GEVernova/After_Aix/AI_method/Models"

OUTPUT_CSV = "/home/yousef/Documents/testing_scripts/GEVernova/After_Aix/AI_method/to_predect/new_acquisition_features.csv"




def load_models_from_folder(model_dir):
    feature_cols_path = os.path.join(model_dir, "feature_columns.joblib")

    if not os.path.exists(feature_cols_path):
        raise FileNotFoundError(f"Fichier introuvable : {feature_cols_path}")

    feature_cols = joblib.load(feature_cols_path)

    models = {}

    for file in os.listdir(model_dir):
        if file.endswith(".joblib") and file != "feature_columns.joblib":
            model_name = file.replace(".joblib", "")
            model_path = os.path.join(model_dir, file)
            models[model_name] = joblib.load(model_path)

    if len(models) == 0:
        raise RuntimeError("Aucun modèle .joblib trouvé dans le dossier Models.")

    print("Features chargées :", len(feature_cols))
    print("Modèles chargés :", list(models.keys()))

    return models, feature_cols




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

    peaks, _ = find_peaks(
        env,
        height=threshold,
        distance=min_distance
    )

    peak_vals = env[peaks] if len(peaks) > 0 else np.array([0.0])

    rms = np.sqrt(np.mean(env ** 2))
    crest = np.max(env) / (rms + 1e-20)

    return {
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

    return {
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
    X = np.fft.fftshift(
        np.fft.fft(blocks * window, axis=1),
        axes=1
    )

    power = np.mean(np.abs(X) ** 2, axis=0)
    power_db = safe_db10(power)

    return {
        "spec_power_mean_db": float(np.mean(power_db)),
        "spec_power_std_db": float(np.std(power_db)),
        "spec_power_max_db": float(np.max(power_db)),
        "spec_power_p95_db": float(np.percentile(power_db, 95)),
        "spec_power_p99_db": float(np.percentile(power_db, 99)),
        "spec_flatness": float(
            np.exp(np.mean(np.log(power + 1e-20)))
            / (np.mean(power) + 1e-20)
        ),
    }




def extract_all_features(samples, freq_center, repeat_idx, rate):
    samples_filt, envelope = preprocess_samples(samples, rate)

    features = {
        "timestamp": datetime.now().isoformat(),
        "label": -1,
        "label_name": "unknown",
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




def acquire_predict_plot_all_models(
    models,
    feature_cols,
    n_repeats=1,
    save_csv=True,
    output_csv=OUTPUT_CSV
):
    print("Initialisation USRP...")
    usrp = uhd.usrp.MultiUSRP()

    freqs = np.arange(F_START, F_STOP + STEP_HZ, STEP_HZ)

    new_features = []

    for rep in range(n_repeats):
        print(f"\n===== Nouvelle acquisition répétition {rep+1}/{n_repeats} =====")

        for i, freq in enumerate(freqs):
            print(f"[{i+1}/{len(freqs)}] Acquisition à {freq/1e6:.1f} MHz")

            samples = acquire_time_domain(
                usrp=usrp,
                freq_center=freq,
                rate=RATE,
                duration=ACQ_DURATION,
                gain=GAIN,
                antenna=ANTENNA
            )

            features = extract_all_features(
                samples=samples,
                freq_center=freq,
                repeat_idx=rep,
                rate=RATE
            )

            new_features.append(features)

            print(
                f"  Peaks={features.get('n_peaks', np.nan)} | "
                f"WavKurtP999={features.get('wav_kurt_p999', np.nan):.2f} | "
                f"Crest={features.get('crest_factor_db', np.nan):.2f} dB"
            )

    df_new = pd.DataFrame(new_features)

    if save_csv:
        df_new.to_csv(output_csv, index=False)
        print("CSV sauvegardé :", output_csv)

    X_new = df_new.reindex(columns=feature_cols)
    X_new = X_new.replace([np.inf, -np.inf], np.nan)

    proba_curves = {}

    plt.figure(figsize=(14, 6))

    for name, model in models.items():
        proba_dp = model.predict_proba(X_new)[:, 1]

        df_tmp = df_new.copy()
        df_tmp["proba_dp"] = proba_dp

        curve = (
            df_tmp
            .groupby("freq_center_mhz")
            .agg(
                proba_dp_mean=("proba_dp", "mean"),
                proba_dp_std=("proba_dp", "std")
            )
            .reset_index()
        )

        proba_curves[name] = curve

        plt.plot(
            curve["freq_center_mhz"],
            100 * curve["proba_dp_mean"],
            label=name
        )

    plt.title("Nouvelle acquisition : probabilité DP en fonction de la fréquence")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("P(DP) (%)")
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df_new, proba_curves




def main():
    models, feature_cols = load_models_from_folder(MODEL_DIR)

    df_new, proba_curves = acquire_predict_plot_all_models(
        models=models,
        feature_cols=feature_cols,
        n_repeats=1,
        save_csv=True,
        output_csv=OUTPUT_CSV
    )

    return df_new, proba_curves


if __name__ == "__main__":
    main()