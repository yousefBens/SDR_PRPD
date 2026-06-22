import numpy as np
import matplotlib.pyplot as plt
import uhd
import time
import json
import pywt

from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis
from datetime import datetime


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

K_THRESHOLD = 8

REF_JSON_PATH = "wavelet_reference_no_dp.json"


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
# OUTILS ROBUSTES
# =====================================================

def robust_median_sigma(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad + 1e-20
    return med, sigma


def robust_threshold(x, K=8):
    med, sigma = robust_median_sigma(x)
    return med + K * sigma


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
# FEATURES WAVELET SUR SIGNAL SANS DP
# =====================================================

def compute_wavelet_reference_features(envelope, rate):
    envelope = np.asarray(envelope, dtype=np.float64).ravel()

    if len(envelope) < WIN_SIZE:
        return None

    env_median, env_sigma = robust_median_sigma(envelope)

    z = (envelope - env_median) / env_sigma

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

    max_med, max_sigma = robust_median_sigma(max_detail)
    energy_med, energy_sigma = robust_median_sigma(energy_detail)
    kurt_med, kurt_sigma = robust_median_sigma(kurt_detail)

    reference = {
        "n_windows": int(len(max_detail)),

        "envelope_median": float(env_median),
        "envelope_sigma": float(env_sigma),

        "max_detail_median": float(max_med),
        "max_detail_sigma": float(max_sigma),
        "max_detail_threshold": float(max_med + K_THRESHOLD * max_sigma),
        "max_detail_p99": float(np.percentile(max_detail, 99)),
        "max_detail_p999": float(np.percentile(max_detail, 99.9)),

        "energy_detail_median": float(energy_med),
        "energy_detail_sigma": float(energy_sigma),
        "energy_detail_threshold": float(energy_med + K_THRESHOLD * energy_sigma),
        "energy_detail_p99": float(np.percentile(energy_detail, 99)),
        "energy_detail_p999": float(np.percentile(energy_detail, 99.9)),

        "kurt_detail_median": float(kurt_med),
        "kurt_detail_sigma": float(kurt_sigma),
        "kurt_detail_threshold": float(kurt_med + K_THRESHOLD * kurt_sigma),
        "kurt_detail_p99": float(np.percentile(kurt_detail, 99)),
        "kurt_detail_p999": float(np.percentile(kurt_detail, 99.9)),
    }

    return reference


# =====================================================
# CALIBRATION SUR TOUTE LA BANDE
# =====================================================

def calibrate_no_dp_reference():
    print("Initialisation USRP...")
    usrp = uhd.usrp.MultiUSRP()

    freqs = np.arange(F_START, F_STOP + STEP_HZ, STEP_HZ)

    all_refs = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "description": "Wavelet reference acquired without DP",
            "f_start_hz": F_START,
            "f_stop_hz": F_STOP,
            "step_hz": STEP_HZ,
            "rate_hz": RATE,
            "gain": GAIN,
            "antenna": ANTENNA,
            "acq_duration_s": ACQ_DURATION,
            "lp_cutoff_hz": LP_CUTOFF_HZ,
            "remove_dc": REMOVE_DC,
            "wavelet": WAVELET_NAME,
            "wavelet_level": WAVELET_LEVEL,
            "win_size": WIN_SIZE,
            "step_win": STEP_WIN,
            "k_threshold": K_THRESHOLD,
        },
        "bands": {}
    }

    for i, freq in enumerate(freqs):
        print(f"\n[{i+1}/{len(freqs)}] Calibration sans DP à {freq/1e6:.1f} MHz")

        samples = acquire_time_domain(
            usrp=usrp,
            freq_center=freq,
            rate=RATE,
            duration=ACQ_DURATION,
            gain=GAIN,
            antenna=ANTENNA
        )

        envelope = preprocess_samples(samples, RATE)

        ref = compute_wavelet_reference_features(envelope, RATE)

        if ref is None:
            print("Signal trop court, référence ignorée.")
            continue

        freq_key = str(int(freq))

        all_refs["bands"][freq_key] = ref

        print(
            f"  Env median={ref['envelope_median']:.4e} | "
            f"Env sigma={ref['envelope_sigma']:.4e}"
        )

        print(
            f"  T_max={ref['max_detail_threshold']:.2f} | "
            f"T_energy={ref['energy_detail_threshold']:.2f} | "
            f"T_kurt={ref['kurt_detail_threshold']:.2f}"
        )

    with open(REF_JSON_PATH, "w") as f:
        json.dump(all_refs, f, indent=4)

    print("\n===== Calibration terminée =====")
    print(f"Fichier JSON enregistré : {REF_JSON_PATH}")

    return all_refs


# =====================================================
# VISUALISATION RÉFÉRENCE
# =====================================================

def plot_reference_json(refs):
    freqs = []
    T_max = []
    T_energy = []
    T_kurt = []

    for freq_key, ref in refs["bands"].items():
        freqs.append(float(freq_key) / 1e6)
        T_max.append(ref["max_detail_threshold"])
        T_energy.append(ref["energy_detail_threshold"])
        T_kurt.append(ref["kurt_detail_threshold"])

    freqs = np.array(freqs)
    T_max = np.array(T_max)
    T_energy = np.array(T_energy)
    T_kurt = np.array(T_kurt)

    plt.figure(figsize=(14, 5))
    plt.plot(freqs, T_max, marker="o")
    plt.title("Référence sans DP : seuil Max Wavelet par fréquence")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Seuil max detail")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(freqs, 10 * np.log10(T_energy + 1e-20), marker="o")
    plt.title("Référence sans DP : seuil énergie Wavelet par fréquence")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Seuil énergie detail (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(freqs, T_kurt, marker="o")
    plt.title("Référence sans DP : seuil Kurtosis Wavelet par fréquence")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Seuil kurtosis detail")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =====================================================
# MAIN
# =====================================================

refs = calibrate_no_dp_reference()
plot_reference_json(refs)