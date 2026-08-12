#!/usr/bin/env python3
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uhd
from scipy.signal import butter, sosfiltfilt


# ============================================================
# 1) CONFIGURATION
# ============================================================

USRP_SERIAL = "306BD15"
CHANNEL = 0
ANTENNA = "RX2"

F_START_HZ = 80e6
F_STOP_HZ = 5.9e9

RATE_HZ = 12e6
GAIN_DB = 40.0

# Pas de scan :
# avec un filtre passe-bas de 4 MHz, un pas de 5 MHz garantit
# qu'un signal se trouve au maximum à 2.5 MHz d'un centre RX.
STEP_HZ = 10e6

# Offset matériel du LO pour réduire les défauts autour de DC.
LO_OFFSET_HZ = 1e6

DISCARD_DURATION_S = 0.002
USEFUL_DURATION_S = 0.020
ACQUISITION_DURATION_S = DISCARD_DURATION_S + USEFUL_DURATION_S

SETTLING_TIME_S = 0.10

# Plusieurs acquisitions par fréquence pour réduire les faux pics.
REPEATS_PER_FREQUENCY = 5

LOWPASS_CUTOFF_HZ = 4e6
FILTER_ORDER = 4

ROBUST_PERCENTILE = 99.99

# Suppression des bords de sosfiltfilt pour éviter les transitoires.
FILTER_EDGE_DURATION_S = 0.0002

CLIPPING_AMPLITUDE = 0.98
EPSILON = 1e-12




# ============================================================
# 2) STRUCTURE DE RÉSULTAT
# ============================================================

@dataclass
class FrequencyResult:
    requested_frequency_hz: float
    actual_frequency_hz: float
    frequency_mhz: float
    gain_db: float

    max_dbfs: float
    robust_dbfs: float
    median_dbfs: float

    max_dbfs_min_repeat: float
    max_dbfs_max_repeat: float
    max_dbfs_std_repeat: float

    peak_amplitude: float
    clipping_fraction: float


# ============================================================
# 3) OUTILS
# ============================================================

def amplitude_to_dbfs(amplitude: float) -> float:
    """
    Convertit une amplitude numérique normalisée en dBFS.
    """
    return float(
        20.0 * np.log10(
            max(float(amplitude), EPSILON)
        )
    )


def create_lowpass_filter(rate_hz: float) -> np.ndarray:
    """
    Crée une seule fois le filtre passe-bas.
    """
    nyquist_hz = rate_hz / 2.0

    if not 0.0 < LOWPASS_CUTOFF_HZ < nyquist_hz:
        raise ValueError(
            f"LOWPASS_CUTOFF_HZ doit être compris entre 0 et "
            f"{nyquist_hz / 1e6:.2f} MHz."
        )

    return butter(
        FILTER_ORDER,
        LOWPASS_CUTOFF_HZ / nyquist_hz,
        btype="low",
        output="sos",
    )


# ============================================================
# 4) CONFIGURATION USRP
# ============================================================

def configure_usrp(
    usrp: uhd.usrp.MultiUSRP,
    center_frequency_hz: float,
) -> tuple[float, float]:
    """
    Configure la fréquence, le gain et l'antenne.
    Retourne la fréquence et le gain réellement appliqués.
    """
    usrp.set_rx_rate(
        RATE_HZ,
        CHANNEL,
    )

    usrp.set_rx_antenna(
        ANTENNA,
        CHANNEL,
    )

    usrp.set_rx_gain(
        GAIN_DB,
        CHANNEL,
    )

    tune_request = uhd.types.TuneRequest(
        center_frequency_hz,
        LO_OFFSET_HZ,
    )

    usrp.set_rx_freq(
        tune_request,
        CHANNEL,
    )

    usrp.set_rx_dc_offset(
        True,
        CHANNEL,
    )

    usrp.set_rx_iq_balance(
        True,
        CHANNEL,
    )

    time.sleep(
        SETTLING_TIME_S
    )

    actual_frequency_hz = float(
        usrp.get_rx_freq(CHANNEL)
    )

    actual_gain_db = float(
        usrp.get_rx_gain(CHANNEL)
    )

    return (
        actual_frequency_hz,
        actual_gain_db,
    )


# ============================================================
# 5) ACQUISITION
# ============================================================

def acquire_time_domain(
    usrp: uhd.usrp.MultiUSRP,
) -> np.ndarray:
    """
    Acquiert un bloc I/Q puis retire le début de l'acquisition.
    """
    number_of_samples = int(
        ACQUISITION_DURATION_S * RATE_HZ
    )

    stream_args = uhd.usrp.StreamArgs(
        "fc32",
        "sc16",
    )
    stream_args.channels = [CHANNEL]

    rx_streamer = usrp.get_rx_stream(
        stream_args
    )

    rx_metadata = uhd.types.RXMetadata()

    samples = np.zeros(
        number_of_samples,
        dtype=np.complex64,
    )

    buffer = np.zeros(
        (
            1,
            rx_streamer.get_max_num_samps(),
        ),
        dtype=np.complex64,
    )

    stream_command = uhd.types.StreamCMD(
        uhd.types.StreamMode.num_done
    )

    stream_command.num_samps = number_of_samples
    stream_command.stream_now = True

    rx_streamer.issue_stream_cmd(
        stream_command
    )

    total_received = 0

    while total_received < number_of_samples:
        received = rx_streamer.recv(
            buffer,
            rx_metadata,
            timeout=3.0,
        )

        if (
            rx_metadata.error_code
            != uhd.types.RXMetadataErrorCode.none
        ):
            raise RuntimeError(
                f"Erreur UHD RX : "
                f"{rx_metadata.strerror()}"
            )

        if received <= 0:
            raise RuntimeError(
                "Aucun échantillon reçu."
            )

        count = min(
            received,
            number_of_samples
            - total_received,
        )

        samples[
            total_received:
            total_received + count
        ] = buffer[0, :count]

        total_received += count

    discard_samples = int(
        DISCARD_DURATION_S * RATE_HZ
    )

    samples = samples[
        discard_samples:
    ]

    if samples.size == 0:
        raise RuntimeError(
            "Aucun échantillon utile après suppression du début."
        )

    return samples


# ============================================================
# 6) CALCUL DES MÉTRIQUES
# ============================================================

def temporal_band_metric(
    samples: np.ndarray,
    lowpass_sos: np.ndarray,
) -> dict:
    """
    Même principe que ton code initial, mais avec :
    - filtre calculé une seule fois ;
    - suppression des bords de filtfilt ;
    - détection du clipping ;
    - gestion stricte des acquisitions vides.
    """
    samples = np.asarray(
        samples,
        dtype=np.complex64,
    ).ravel()

    if samples.size == 0:
        raise ValueError(
            "Impossible de traiter une acquisition vide."
        )

    filtered_samples = sosfiltfilt(
        lowpass_sos,
        samples,
    )

    edge_samples = int(
        FILTER_EDGE_DURATION_S * RATE_HZ
    )

    if (
        edge_samples > 0
        and filtered_samples.size > 2 * edge_samples
    ):
        filtered_samples = filtered_samples[
            edge_samples:
            -edge_samples
        ]

    envelope = np.abs(
        filtered_samples
    )

    max_amplitude = float(
        np.max(envelope)
    )

    robust_amplitude = float(
        np.percentile(
            envelope,
            ROBUST_PERCENTILE,
        )
    )

    median_amplitude = float(
        np.median(envelope)
    )

    clipping_fraction = float(
        np.mean(
            np.abs(samples)
            >= CLIPPING_AMPLITUDE
        )
    )

    return {
        "max_amplitude": max_amplitude,
        "max_dbfs": amplitude_to_dbfs(
            max_amplitude
        ),
        "robust_dbfs": amplitude_to_dbfs(
            robust_amplitude
        ),
        "median_dbfs": amplitude_to_dbfs(
            median_amplitude
        ),
        "clipping_fraction": (
            clipping_fraction
        ),
    }


def measure_frequency_window(
    usrp: uhd.usrp.MultiUSRP,
    lowpass_sos: np.ndarray,
) -> dict:
    """
    Réalise plusieurs acquisitions à la même fréquence.

    La valeur principale est la médiane des maxima temporels.
    Elle est plus fiable qu'un seul maximum.
    """
    measurements = []

    for _ in range(
        REPEATS_PER_FREQUENCY
    ):
        samples = acquire_time_domain(
            usrp
        )

        measurements.append(
            temporal_band_metric(
                samples,
                lowpass_sos,
            )
        )

    max_dbfs_values = np.asarray(
        [
            measurement["max_dbfs"]
            for measurement in measurements
        ],
        dtype=float,
    )

    robust_dbfs_values = np.asarray(
        [
            measurement["robust_dbfs"]
            for measurement in measurements
        ],
        dtype=float,
    )

    median_dbfs_values = np.asarray(
        [
            measurement["median_dbfs"]
            for measurement in measurements
        ],
        dtype=float,
    )

    return {
        "max_dbfs": float(
            np.median(
                max_dbfs_values
            )
        ),

        "robust_dbfs": float(
            np.median(
                robust_dbfs_values
            )
        ),

        "median_dbfs": float(
            np.median(
                median_dbfs_values
            )
        ),

        "max_dbfs_min_repeat": float(
            np.min(
                max_dbfs_values
            )
        ),

        "max_dbfs_max_repeat": float(
            np.max(
                max_dbfs_values
            )
        ),

        "max_dbfs_std_repeat": float(
            np.std(
                max_dbfs_values
            )
        ),

        "peak_amplitude": float(
            np.max(
                [
                    measurement[
                        "max_amplitude"
                    ]
                    for measurement in measurements
                ]
            )
        ),

        "clipping_fraction": float(
            np.max(
                [
                    measurement[
                        "clipping_fraction"
                    ]
                    for measurement in measurements
                ]
            )
        ),
    }


# ============================================================
# 7) SCAN FRÉQUENTIEL
# ============================================================

def scan_pd_time_domain() -> list[FrequencyResult]:
    print(
        "Initialisation USRP..."
    )

    usrp = uhd.usrp.MultiUSRP(
        f"serial={USRP_SERIAL}"
    )

    lowpass_sos = create_lowpass_filter(
        RATE_HZ
    )

    frequencies_hz = np.arange(
        F_START_HZ,
        F_STOP_HZ
        + STEP_HZ / 2.0,
        STEP_HZ,
        dtype=float,
    )

    print()
    print(
        "Configure manuellement ton générateur "
        "ou ta source de pulses."
    )
    print(
        "Laisse l'injection active pendant tout le scan."
    )

    input(
        "Appuie sur Entrée lorsque l'injection est prête..."
    )

    results = []

    for index, requested_frequency_hz in enumerate(
        frequencies_hz,
        start=1,
    ):
        (
            actual_frequency_hz,
            actual_gain_db,
        ) = configure_usrp(
            usrp,
            float(
                requested_frequency_hz
            ),
        )

        metrics = measure_frequency_window(
            usrp,
            lowpass_sos,
        )

        result = FrequencyResult(
            requested_frequency_hz=float(
                requested_frequency_hz
            ),

            actual_frequency_hz=float(
                actual_frequency_hz
            ),

            frequency_mhz=float(
                actual_frequency_hz / 1e6
            ),

            gain_db=float(
                actual_gain_db
            ),

            max_dbfs=float(
                metrics["max_dbfs"]
            ),

            robust_dbfs=float(
                metrics["robust_dbfs"]
            ),

            median_dbfs=float(
                metrics["median_dbfs"]
            ),

            max_dbfs_min_repeat=float(
                metrics[
                    "max_dbfs_min_repeat"
                ]
            ),

            max_dbfs_max_repeat=float(
                metrics[
                    "max_dbfs_max_repeat"
                ]
            ),

            max_dbfs_std_repeat=float(
                metrics[
                    "max_dbfs_std_repeat"
                ]
            ),

            peak_amplitude=float(
                metrics[
                    "peak_amplitude"
                ]
            ),

            clipping_fraction=float(
                metrics[
                    "clipping_fraction"
                ]
            ),
        )

        results.append(
            result
        )

        print(
            f"[{index:3d}/"
            f"{len(frequencies_hz)}] "
            f"{result.frequency_mhz:8.1f} MHz | "
            f"Max={result.max_dbfs:7.2f} dBFS | "
            f"Robuste={result.robust_dbfs:7.2f} dBFS | "
            f"Écart répétitions="
            f"{result.max_dbfs_std_repeat:.2f} dB | "
            f"Clip="
            f"{result.clipping_fraction:.2e}"
        )

    return results



# ============================================================
# 9) AFFICHAGE
# ============================================================

def plot_temporal_scan(
    results: list[FrequencyResult],
) -> None:
    frequencies_mhz = np.asarray(
        [
            result.frequency_mhz
            for result in results
        ],
        dtype=float,
    )

    max_values = np.asarray(
        [
            result.max_dbfs
            for result in results
        ],
        dtype=float,
    )

    robust_values = np.asarray(
        [
            result.robust_dbfs
            for result in results
        ],
        dtype=float,
    )

    min_repeat = np.asarray(
        [
            result.max_dbfs_min_repeat
            for result in results
        ],
        dtype=float,
    )

    max_repeat = np.asarray(
        [
            result.max_dbfs_max_repeat
            for result in results
        ],
        dtype=float,
    )

    plt.figure(
        figsize=(15, 7)
    )

    plt.fill_between(
        frequencies_mhz,
        min_repeat,
        max_repeat,
        alpha=0.18,
        label=(
            "Variation du MAX "
            "entre acquisitions"
        ),
    )

    plt.plot(
        frequencies_mhz,
        max_values,
        linewidth=1.7,
        label=(
            "Médiane du MAX temporel "
            "par fenêtre"
        ),
    )

    plt.plot(
        frequencies_mhz,
        robust_values,
        linewidth=1.5,
        label=(
            f"Percentile "
            f"{ROBUST_PERCENTILE}%"
        ),
    )

    plt.title(
        f"Détection de pulses par scan temporel — "
        f"gain RX = {GAIN_DB:.0f} dB"
    )

    plt.xlabel(
        "Fréquence centrale RX (MHz)"
    )

    plt.ylabel(
        "Amplitude détectée (dBFS)"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()



    plt.show()


# ============================================================
# 10) MAIN
# ============================================================

def main() -> None:
    start_time = time.perf_counter()

    results = scan_pd_time_domain()



    plot_temporal_scan(
        results
    )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        f"Durée totale : "
        f"{elapsed_time:.1f} s"
    )



if __name__ == "__main__":
    main()