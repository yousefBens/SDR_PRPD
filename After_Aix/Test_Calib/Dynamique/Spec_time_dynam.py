#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uhd

USRP_SERIAL = "306BD15"
CHANNEL = 0
ANTENNA = "RX2"
F_START_HZ = 100e6
F_STOP_HZ = 2e9
STEP_HZ = 6e6
RATE_HZ = 12e6
GAIN_DB = 40.0
ACQUISITION_DURATION_S = 0.020
DISCARD_DURATION_S = 0.002
SETTLING_TIME_S = 0.10
LO_OFFSET_HZ = 2e6
REPEATS_PER_FREQUENCY = 3
ROBUST_PERCENTILE = 99.99
PULSE_THRESHOLD_MAD = 8.0
MIN_PULSE_SAMPLES = 3
MIN_TOTAL_EXCESS_DB = 3.0
MIN_PULSE_EXCESS_DB = 6.0
CLIPPING_AMPLITUDE = 0.98
EPSILON = 1e-20
OUTPUT_DIR = Path.home() / "b200_pd_spectrum_scan"
CSV_FILE = OUTPUT_DIR / "pd_spectrum_results.csv"
PLOT_FILE = OUTPUT_DIR / "pd_spectrum.png"
PLOT_DETAILED_FILE = OUTPUT_DIR / "pd_spectrum_detailed.png"

@dataclass
class FrequencyResult:
    frequency_hz: float
    frequency_mhz: float
    gain_db: float
    noise_total_dbfs: float
    noise_robust_dbfs: float
    noise_median_amplitude_dbfs: float
    signal_total_dbfs: float
    signal_robust_dbfs: float
    signal_median_amplitude_dbfs: float
    signal_max_amplitude_dbfs: float
    total_excess_db: float
    pulse_excess_db: float
    crest_factor_db: float
    detected_pulse_samples: int
    pulse_probability: float
    peak_amplitude: float
    clipping_fraction: float
    detected: bool


def power_to_dbfs(power_linear: float) -> float:
    return float(10.0 * math.log10(max(float(power_linear), EPSILON)))


def amplitude_to_dbfs(amplitude: float) -> float:
    return float(20.0 * math.log10(max(float(amplitude), math.sqrt(EPSILON))))


def configure_usrp(usrp, center_frequency_hz: float) -> tuple[float, float]:
    usrp.set_rx_rate(RATE_HZ, CHANNEL)
    usrp.set_rx_antenna(ANTENNA, CHANNEL)
    usrp.set_rx_gain(GAIN_DB, CHANNEL)
    usrp.set_rx_freq(uhd.types.TuneRequest(center_frequency_hz, LO_OFFSET_HZ), CHANNEL)
    usrp.set_rx_dc_offset(True, CHANNEL)
    usrp.set_rx_iq_balance(True, CHANNEL)
    time.sleep(SETTLING_TIME_S)
    return float(usrp.get_rx_freq(CHANNEL)), float(usrp.get_rx_gain(CHANNEL))


def acquire_samples(usrp) -> np.ndarray:
    number_of_samples = int((ACQUISITION_DURATION_S + DISCARD_DURATION_S) * RATE_HZ)
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [CHANNEL]
    streamer = usrp.get_rx_stream(stream_args)
    metadata = uhd.types.RXMetadata()
    samples = np.zeros(number_of_samples, dtype=np.complex64)
    buffer = np.zeros((1, streamer.get_max_num_samps()), dtype=np.complex64)
    command = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    command.num_samps = number_of_samples
    command.stream_now = True
    streamer.issue_stream_cmd(command)
    total_received = 0
    while total_received < number_of_samples:
        received = streamer.recv(buffer, metadata, timeout=3.0)
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            raise RuntimeError(f"Erreur UHD RX : {metadata.strerror()}")
        if received <= 0:
            raise RuntimeError("Aucun échantillon reçu.")
        count = min(received, number_of_samples - total_received)
        samples[total_received:total_received + count] = buffer[0, :count]
        total_received += count
    discarded_samples = int(DISCARD_DURATION_S * RATE_HZ)
    return samples[discarded_samples:]


def calculate_metrics(samples: np.ndarray) -> dict:
    samples = np.asarray(samples, dtype=np.complex64).ravel()
    if samples.size == 0:
        raise ValueError("Acquisition vide.")
    magnitude = np.abs(samples)
    total_power_dbfs = power_to_dbfs(float(np.mean(magnitude ** 2)))
    median_amplitude = float(np.median(magnitude))
    robust_amplitude = float(np.percentile(magnitude, ROBUST_PERCENTILE))
    max_amplitude = float(np.max(magnitude))
    median_amplitude_dbfs = amplitude_to_dbfs(median_amplitude)
    robust_amplitude_dbfs = amplitude_to_dbfs(robust_amplitude)
    max_amplitude_dbfs = amplitude_to_dbfs(max_amplitude)
    mad = float(np.median(np.abs(magnitude - median_amplitude)))
    robust_sigma = 1.4826 * mad
    pulse_threshold = median_amplitude + PULSE_THRESHOLD_MAD * max(robust_sigma, math.sqrt(EPSILON))
    pulse_mask = magnitude > pulse_threshold
    pulse_count = int(np.count_nonzero(pulse_mask))
    pulse_probability = float(pulse_count / magnitude.size)
    crest_factor_db = max_amplitude_dbfs - total_power_dbfs
    clipping_fraction = float(np.mean(magnitude >= CLIPPING_AMPLITUDE))
    return {
        "total_power_dbfs": total_power_dbfs,
        "robust_amplitude_dbfs": robust_amplitude_dbfs,
        "median_amplitude_dbfs": median_amplitude_dbfs,
        "max_amplitude_dbfs": max_amplitude_dbfs,
        "crest_factor_db": crest_factor_db,
        "pulse_count": pulse_count,
        "pulse_probability": pulse_probability,
        "peak_amplitude": max_amplitude,
        "clipping_fraction": clipping_fraction,
    }


def measure_frequency(usrp) -> dict:
    measurements = [calculate_metrics(acquire_samples(usrp)) for _ in range(REPEATS_PER_FREQUENCY)]
    return {
        "total_power_dbfs": float(np.median([m["total_power_dbfs"] for m in measurements])),
        "robust_amplitude_dbfs": float(np.median([m["robust_amplitude_dbfs"] for m in measurements])),
        "median_amplitude_dbfs": float(np.median([m["median_amplitude_dbfs"] for m in measurements])),
        "max_amplitude_dbfs": float(np.max([m["max_amplitude_dbfs"] for m in measurements])),
        "crest_factor_db": float(np.median([m["crest_factor_db"] for m in measurements])),
        "pulse_count": int(np.max([m["pulse_count"] for m in measurements])),
        "pulse_probability": float(np.max([m["pulse_probability"] for m in measurements])),
        "peak_amplitude": float(np.max([m["peak_amplitude"] for m in measurements])),
        "clipping_fraction": float(np.max([m["clipping_fraction"] for m in measurements])),
    }


def scan_condition(usrp, frequencies_hz: np.ndarray, label: str) -> list[dict]:
    results = []
    print("\n" + "=" * 76)
    print(label)
    print("=" * 76)
    for index, frequency_hz in enumerate(frequencies_hz, start=1):
        actual_frequency, actual_gain = configure_usrp(usrp, float(frequency_hz))
        metrics = measure_frequency(usrp)
        results.append({
            "frequency_hz": actual_frequency,
            "frequency_mhz": actual_frequency / 1e6,
            "gain_db": actual_gain,
            **metrics,
        })
        print(
            f"[{index:3d}/{len(frequencies_hz)}] {actual_frequency/1e6:8.1f} MHz | "
            f"Total={metrics['total_power_dbfs']:7.2f} dBFS | "
            f"Robuste={metrics['robust_amplitude_dbfs']:7.2f} dBFS | "
            f"Max={metrics['max_amplitude_dbfs']:7.2f} dBFS | "
            f"Pulses={metrics['pulse_count']:5d} | Clip={metrics['clipping_fraction']:.2e}"
        )
    return results


def build_final_results(noise_results: list[dict], signal_results: list[dict]) -> list[FrequencyResult]:
    if len(noise_results) != len(signal_results):
        raise ValueError("Les deux scans n'ont pas le même nombre de fréquences.")
    final_results = []
    for noise, signal in zip(noise_results, signal_results):
        if not np.isclose(noise["frequency_hz"], signal["frequency_hz"], atol=1.0):
            raise ValueError("Les fréquences des deux scans ne correspondent pas.")
        total_excess_db = signal["total_power_dbfs"] - noise["total_power_dbfs"]
        pulse_excess_db = signal["robust_amplitude_dbfs"] - noise["robust_amplitude_dbfs"]
        detected = bool(
            signal["pulse_count"] >= MIN_PULSE_SAMPLES
            and (total_excess_db >= MIN_TOTAL_EXCESS_DB or pulse_excess_db >= MIN_PULSE_EXCESS_DB)
        )
        final_results.append(FrequencyResult(
            frequency_hz=signal["frequency_hz"],
            frequency_mhz=signal["frequency_mhz"],
            gain_db=signal["gain_db"],
            noise_total_dbfs=noise["total_power_dbfs"],
            noise_robust_dbfs=noise["robust_amplitude_dbfs"],
            noise_median_amplitude_dbfs=noise["median_amplitude_dbfs"],
            signal_total_dbfs=signal["total_power_dbfs"],
            signal_robust_dbfs=signal["robust_amplitude_dbfs"],
            signal_median_amplitude_dbfs=signal["median_amplitude_dbfs"],
            signal_max_amplitude_dbfs=signal["max_amplitude_dbfs"],
            total_excess_db=total_excess_db,
            pulse_excess_db=pulse_excess_db,
            crest_factor_db=signal["crest_factor_db"],
            detected_pulse_samples=signal["pulse_count"],
            pulse_probability=signal["pulse_probability"],
            peak_amplitude=signal["peak_amplitude"],
            clipping_fraction=signal["clipping_fraction"],
            detected=detected,
        ))
    return final_results





def plot_results(results: list[FrequencyResult]) -> None:
    frequencies = np.asarray([r.frequency_mhz for r in results], dtype=float)
    noise_total = np.asarray([r.noise_total_dbfs for r in results], dtype=float)
    signal_total = np.asarray([r.signal_total_dbfs for r in results], dtype=float)
    noise_robust = np.asarray([r.noise_robust_dbfs for r in results], dtype=float)
    signal_robust = np.asarray([r.signal_robust_dbfs for r in results], dtype=float)
    detected = np.asarray([r.detected for r in results], dtype=bool)

    plt.figure(figsize=(14, 7))
    plt.plot(frequencies, noise_total, linewidth=2.2, label="Noise floor — puissance moyenne totale")
    plt.plot(frequencies, signal_total, linewidth=1.8, label="Avec DP — puissance moyenne totale")
    plt.plot(frequencies, signal_robust, linewidth=1.3, label=f"Avec DP — percentile {ROBUST_PERCENTILE} %")
    if np.any(detected):
        plt.scatter(frequencies[detected], signal_robust[detected], marker="x", s=55, label="Impulsions détectées")
    plt.title(f"Scan spectral temporel du B200 — gain RX = {GAIN_DB:.0f} dB")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Niveau numérique (dBFS)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200)
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(frequencies, noise_total, linewidth=2.2, label="Noise floor total")
    axes[0].plot(frequencies, signal_total, linewidth=1.8, label="Puissance totale avec DP")
    axes[0].set_title("Puissance moyenne large bande — comparable au grand script")
    axes[0].set_ylabel("Puissance totale (dBFS)")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(frequencies, noise_robust, linewidth=2.0, label=f"Bruit — percentile {ROBUST_PERCENTILE} %")
    axes[1].plot(frequencies, signal_robust, linewidth=1.8, label=f"Avec DP — percentile {ROBUST_PERCENTILE} %")
    if np.any(detected):
        axes[1].scatter(frequencies[detected], signal_robust[detected], marker="x", s=55, label="Impulsions détectées")
    axes[1].set_title("Métrique temporelle robuste pour impulsions courtes")
    axes[1].set_xlabel("Fréquence centrale RX (MHz)")
    axes[1].set_ylabel("Amplitude robuste (dBFS)")
    axes[1].grid(True)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(PLOT_DETAILED_FILE, dpi=200)
    plt.show()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frequencies_hz = np.arange(F_START_HZ, F_STOP_HZ + STEP_HZ / 2.0, STEP_HZ, dtype=float)
    print("Initialisation du B200...")
    usrp = uhd.usrp.MultiUSRP(f"serial={USRP_SERIAL}")

    print("\nÉtape 1 : mets le banc SANS DP, puis appuie sur Entrée.")
    input()
    noise_results = scan_condition(usrp, frequencies_hz, "SCAN DE RÉFÉRENCE — SANS DP")

    print("\nÉtape 2 : active les DP, puis appuie sur Entrée.")
    input()
    signal_results = scan_condition(usrp, frequencies_hz, "SCAN COURANT — AVEC DP")

    final_results = build_final_results(noise_results, signal_results)
    plot_results(final_results)

    detected_frequencies = [r.frequency_mhz for r in final_results if r.detected]
    print("\nScan terminé.")
    print(f"CSV    : {CSV_FILE}")
    print(f"Graphe : {PLOT_FILE}")
    print(f"Détail : {PLOT_DETAILED_FILE}")
    if detected_frequencies:
        print("Bandes avec impulsions détectées (MHz) :")
        print(", ".join(f"{f:.1f}" for f in detected_frequencies))
    else:
        print("Aucune bande ne satisfait les critères de détection configurés.")


if __name__ == "__main__":
    main()