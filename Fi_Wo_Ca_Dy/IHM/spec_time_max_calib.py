#!/usr/bin/env python3
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uhd
from scipy.signal import butter, sosfiltfilt


# ============================================================
# 1) CONFIGURATION USRP ET SCAN
# ============================================================

USRP_SERIAL = "306BD15"
CHANNEL = 0
ANTENNA = "RX2"

F_START_HZ = 100e6
F_STOP_HZ = 2.0e9

RATE_HZ = 12e6
GAIN_DB = 40.0

STEP_HZ = 12e6

# Offset matériel du LO afin de limiter les défauts autour de DC.
LO_OFFSET_HZ = 1e6

DISCARD_DURATION_S = 0.001
USEFUL_DURATION_S = 0.021

ACQUISITION_DURATION_S = (
    DISCARD_DURATION_S
    + USEFUL_DURATION_S
)

SETTLING_TIME_S = 0.10

REPEATS_PER_FREQUENCY = 1

LOWPASS_CUTOFF_HZ = 5.99e6
FILTER_ORDER = 4

ROBUST_PERCENTILE = 99.99

FILTER_EDGE_DURATION_S = 0.0002

CLIPPING_AMPLITUDE = 0.98
EPSILON = 1e-12


# ============================================================
# 2) TA TABLE DE CALIBRATION UHD
# ============================================================

CAL_FILE = Path(
    "/home/yousef/Documents/testing_scripts/GEVernova/"
    "Fi_Wo_Ca_Dy/Calib/Cal_Files/"
    "rx2_power_calibration_final.pickle"
)

# Tolérance maximale entre le gain demandé et un gain calibré.
MAX_GAIN_DISTANCE_DB = 1.0

# Petite tolérance aux limites fréquentielles UHD.
CAL_FREQUENCY_TOLERANCE_HZ = 1e3


# ============================================================
# 3) STRUCTURE DE RÉSULTAT
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

    calibration_reference_dbm: float

    max_estimated_dbm: float
    robust_estimated_dbm: float
    median_estimated_dbm: float

    max_dbfs_min_repeat: float
    max_dbfs_max_repeat: float
    max_dbfs_std_repeat: float

    peak_amplitude: float
    clipping_fraction: float


# ============================================================
# 4) OUTILS dBFS
# ============================================================

def amplitude_to_dbfs(
    amplitude: float,
) -> float:
    """
    Pour une amplitude numérique normalisée :

        dBFS = 20 log10(amplitude)
    """

    return float(
        20.0
        * np.log10(
            max(
                float(amplitude),
                EPSILON,
            )
        )
    )


def create_lowpass_filter(
    rate_hz: float,
) -> np.ndarray:

    nyquist_hz = (
        rate_hz
        / 2.0
    )

    if not (
        0.0
        < LOWPASS_CUTOFF_HZ
        < nyquist_hz
    ):
        raise ValueError(
            "LOWPASS_CUTOFF_HZ doit être compris entre 0 et "
            f"{nyquist_hz / 1e6:.2f} MHz."
        )

    return butter(
        FILTER_ORDER,
        LOWPASS_CUTOFF_HZ / nyquist_hz,
        btype="low",
        output="sos",
    )


# ============================================================
# 5) CHARGEMENT DE TA CALIBRATION PICKLE
# ============================================================

def load_power_calibration(
    calibration_file: Path,
) -> dict:
    """
    Charge :

        rx2_power_calibration_final.pickle

    Structure attendue :

        calibration[0]["RX2"][frequency_hz][gain_db]
            = reference_dbm

    La conversion utilisée ensuite est :

        P_RX2_dBm = P_dBFS + reference_dBm
    """

    if not calibration_file.exists():
        raise FileNotFoundError(
            "Fichier de calibration introuvable :\n"
            f"{calibration_file}"
        )

    with calibration_file.open(
        "rb"
    ) as file:
        calibration = pickle.load(
            file
        )

    if 0 not in calibration:
        raise ValueError(
            "Canal 0 absent de la table de calibration."
        )

    if ANTENNA not in calibration[0]:
        raise ValueError(
            f"Antenne {ANTENNA} absente de la calibration."
        )

    raw_table = calibration[
        0
    ][
        ANTENNA
    ]

    if not raw_table:
        raise ValueError(
            "La table de calibration est vide."
        )

    # Normalisation des clés en float.
    table = {}

    for frequency_hz, gain_table in raw_table.items():

        frequency_hz = float(
            frequency_hz
        )

        table[
            frequency_hz
        ] = {
            float(gain_db):
                float(reference_dbm)

            for gain_db, reference_dbm
            in gain_table.items()
        }

    return table


def select_calibrated_gain(
    calibration_table: dict,
    requested_gain_db: float,
) -> float:
    """
    Sélectionne le gain calibré le plus proche.

    Exemple :
        gains disponibles :
        0, 20, 40, 60, 76 dB
    """

    gains = sorted(
        {
            float(gain_db)

            for gain_table
            in calibration_table.values()

            for gain_db
            in gain_table.keys()
        }
    )

    if not gains:
        raise ValueError(
            "Aucun gain présent dans la calibration."
        )

    gains_array = np.asarray(
        gains,
        dtype=float,
    )

    selected_gain = float(
        gains_array[
            np.argmin(
                np.abs(
                    gains_array
                    - requested_gain_db
                )
            )
        ]
    )

    distance = abs(
        selected_gain
        - requested_gain_db
    )

    if distance > MAX_GAIN_DISTANCE_DB:

        raise ValueError(
            f"Gain demandé : "
            f"{requested_gain_db:.1f} dB\n"
            f"Gain calibré le plus proche : "
            f"{selected_gain:.1f} dB\n"
            f"Écart : {distance:.1f} dB"
        )

    return selected_gain


def build_gain_calibration(
    calibration_table: dict,
    gain_db: float,
) -> dict:
    """
    Extrait toutes les références (fréquence -> référence dBm)
    pour le gain demandé.
    """

    selected_gain = select_calibrated_gain(
        calibration_table,
        gain_db,
    )

    frequencies = []
    references = []

    for frequency_hz in sorted(
        calibration_table.keys()
    ):

        gain_table = calibration_table[
            frequency_hz
        ]

        if selected_gain not in gain_table:
            continue

        frequencies.append(
            float(
                frequency_hz
            )
        )

        references.append(
            float(
                gain_table[
                    selected_gain
                ]
            )
        )

    if not frequencies:

        raise ValueError(
            f"Aucun point de calibration pour "
            f"le gain {selected_gain:.1f} dB."
        )

    return {
        "gain_db":
            float(selected_gain),

        "frequencies_hz":
            np.asarray(
                frequencies,
                dtype=float,
            ),

        "references_dbm":
            np.asarray(
                references,
                dtype=float,
            ),
    }


def interpolate_reference_dbm(
    gain_calibration: dict,
    frequency_hz: float,
) -> float:
    """
    Retourne la référence dBm à utiliser pour une fréquence.

    Si la fréquence n'est pas exactement dans la table,
    interpolation linéaire entre les deux références voisines.
    """

    frequency_hz = float(
        frequency_hz
    )

    frequencies = gain_calibration[
        "frequencies_hz"
    ]

    references = gain_calibration[
        "references_dbm"
    ]

    minimum_frequency_hz = float(
        frequencies.min()
    )

    maximum_frequency_hz = float(
        frequencies.max()
    )

    # Gestion des petites différences numériques UHD.
    if frequency_hz < minimum_frequency_hz:

        difference_hz = (
            minimum_frequency_hz
            - frequency_hz
        )

        if (
            difference_hz
            <= CAL_FREQUENCY_TOLERANCE_HZ
        ):
            frequency_hz = (
                minimum_frequency_hz
            )

        else:
            raise ValueError(
                f"Fréquence "
                f"{frequency_hz / 1e6:.6f} MHz "
                "inférieure à la calibration "
                f"({minimum_frequency_hz / 1e6:.1f} MHz)."
            )

    if frequency_hz > maximum_frequency_hz:

        difference_hz = (
            frequency_hz
            - maximum_frequency_hz
        )

        if (
            difference_hz
            <= CAL_FREQUENCY_TOLERANCE_HZ
        ):
            frequency_hz = (
                maximum_frequency_hz
            )

        else:
            raise ValueError(
                f"Fréquence "
                f"{frequency_hz / 1e6:.6f} MHz "
                "supérieure à la calibration "
                f"({maximum_frequency_hz / 1e6:.1f} MHz)."
            )

    reference_dbm = np.interp(
        frequency_hz,
        frequencies,
        references,
    )

    return float(
        reference_dbm
    )


def dbfs_to_dbm(
    gain_calibration: dict,
    measured_dbfs: float,
    frequency_hz: float,
) -> tuple[float, float]:
    """
    Conversion avec ta table finale UHD :

        dBm = dBFS + référence(fréquence, gain)

    Retourne :
        estimated_dbm
        reference_dbm
    """

    reference_dbm = (
        interpolate_reference_dbm(
            gain_calibration,
            frequency_hz,
        )
    )

    estimated_dbm = (
        float(measured_dbfs)
        + reference_dbm
    )

    return (
        float(estimated_dbm),
        float(reference_dbm),
    )


# ============================================================
# 6) CONFIGURATION USRP
# ============================================================

def configure_usrp(
    usrp: uhd.usrp.MultiUSRP,
    center_frequency_hz: float,
) -> tuple[float, float]:

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

    tune_request = (
        uhd.types.TuneRequest(
            center_frequency_hz,
            LO_OFFSET_HZ,
        )
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
        usrp.get_rx_freq(
            CHANNEL
        )
    )

    actual_gain_db = float(
        usrp.get_rx_gain(
            CHANNEL
        )
    )

    return (
        actual_frequency_hz,
        actual_gain_db,
    )


# ============================================================
# 7) ACQUISITION
# ============================================================

def acquire_time_domain(
    usrp: uhd.usrp.MultiUSRP,
) -> np.ndarray:

    number_of_samples = int(
        ACQUISITION_DURATION_S
        * RATE_HZ
    )

    stream_args = (
        uhd.usrp.StreamArgs(
            "fc32",
            "sc16",
        )
    )

    stream_args.channels = [
        CHANNEL
    ]

    rx_streamer = (
        usrp.get_rx_stream(
            stream_args
        )
    )

    rx_metadata = (
        uhd.types.RXMetadata()
    )

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

    stream_command = (
        uhd.types.StreamCMD(
            uhd.types.StreamMode.num_done
        )
    )

    stream_command.num_samps = (
        number_of_samples
    )

    stream_command.stream_now = True

    rx_streamer.issue_stream_cmd(
        stream_command
    )

    total_received = 0

    while (
        total_received
        < number_of_samples
    ):

        received = (
            rx_streamer.recv(
                buffer,
                rx_metadata,
                timeout=3.0,
            )
        )

        if (
            rx_metadata.error_code
            !=
            uhd.types.RXMetadataErrorCode.none
        ):
            raise RuntimeError(
                "Erreur UHD RX : "
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
        ] = buffer[
            0,
            :count
        ]

        total_received += count

    discard_samples = int(
        DISCARD_DURATION_S
        * RATE_HZ
    )

    samples = samples[
        discard_samples:
    ]

    if samples.size == 0:
        raise RuntimeError(
            "Aucun échantillon utile."
        )

    return samples


# ============================================================
# 8) CALCUL MAX TEMPOREL
# ============================================================

def temporal_band_metric(
    samples: np.ndarray,
    lowpass_sos: np.ndarray,
) -> dict:

    samples = np.asarray(
        samples,
        dtype=np.complex64,
    ).ravel()

    if samples.size == 0:
        raise ValueError(
            "Acquisition vide."
        )

    filtered_samples = sosfiltfilt(
        lowpass_sos,
        samples,
    )

    edge_samples = int(
        FILTER_EDGE_DURATION_S
        * RATE_HZ
    )

    if (
        edge_samples > 0
        and filtered_samples.size
        > 2 * edge_samples
    ):

        filtered_samples = (
            filtered_samples[
                edge_samples:
                -edge_samples
            ]
        )

    envelope = np.abs(
        filtered_samples
    )

    max_amplitude = float(
        np.max(
            envelope
        )
    )

    robust_amplitude = float(
        np.percentile(
            envelope,
            ROBUST_PERCENTILE,
        )
    )

    median_amplitude = float(
        np.median(
            envelope
        )
    )

    clipping_fraction = float(
        np.mean(
            np.abs(
                samples
            )
            >= CLIPPING_AMPLITUDE
        )
    )

    return {
        "max_amplitude":
            max_amplitude,

        "max_dbfs":
            amplitude_to_dbfs(
                max_amplitude
            ),

        "robust_dbfs":
            amplitude_to_dbfs(
                robust_amplitude
            ),

        "median_dbfs":
            amplitude_to_dbfs(
                median_amplitude
            ),

        "clipping_fraction":
            clipping_fraction,
    }


def measure_frequency_window(
    usrp: uhd.usrp.MultiUSRP,
    lowpass_sos: np.ndarray,
) -> dict:

    measurements = []

    for _ in range(
        REPEATS_PER_FREQUENCY
    ):

        samples = (
            acquire_time_domain(
                usrp
            )
        )

        measurements.append(
            temporal_band_metric(
                samples,
                lowpass_sos,
            )
        )

    max_dbfs_values = np.asarray(
        [
            measurement[
                "max_dbfs"
            ]
            for measurement
            in measurements
        ],
        dtype=float,
    )

    robust_dbfs_values = np.asarray(
        [
            measurement[
                "robust_dbfs"
            ]
            for measurement
            in measurements
        ],
        dtype=float,
    )

    median_dbfs_values = np.asarray(
        [
            measurement[
                "median_dbfs"
            ]
            for measurement
            in measurements
        ],
        dtype=float,
    )

    return {
        "max_dbfs":
            float(
                np.median(
                    max_dbfs_values
                )
            ),

        "robust_dbfs":
            float(
                np.median(
                    robust_dbfs_values
                )
            ),

        "median_dbfs":
            float(
                np.median(
                    median_dbfs_values
                )
            ),

        "max_dbfs_min_repeat":
            float(
                np.min(
                    max_dbfs_values
                )
            ),

        "max_dbfs_max_repeat":
            float(
                np.max(
                    max_dbfs_values
                )
            ),

        "max_dbfs_std_repeat":
            float(
                np.std(
                    max_dbfs_values
                )
            ),

        "peak_amplitude":
            float(
                np.max(
                    [
                        measurement[
                            "max_amplitude"
                        ]
                        for measurement
                        in measurements
                    ]
                )
            ),

        "clipping_fraction":
            float(
                np.max(
                    [
                        measurement[
                            "clipping_fraction"
                        ]
                        for measurement
                        in measurements
                    ]
                )
            ),
    }


# ============================================================
# 9) SCAN FRÉQUENTIEL
# ============================================================

def scan_pd_time_domain(
    gain_calibration: dict,
) -> tuple[list[FrequencyResult], float, dict]:

    print(
        "Initialisation USRP..."
    )

    usrp = uhd.usrp.MultiUSRP(
        f"serial={USRP_SERIAL}"
    )

    lowpass_sos = (
        create_lowpass_filter(
            RATE_HZ
        )
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
        "Configure ta source de signal/pulses."
    )

    print(
        "Laisse l'injection active pendant tout le scan."
    )

    input(
        "Appuie sur Entrée lorsque l'injection est prête..."
    )

    # Début du chronométrage :
    # scan + acquisition + traitement + calibration + affichage.
    scan_start_time = time.perf_counter()

    timing = {
        "tuning_configuration_s": 0.0,
        "acquisition_s": 0.0,
        "processing_s": 0.0,
        "calibration_s": 0.0,
    }

    results = []

    for index, requested_frequency_hz in enumerate(
        frequencies_hz,
        start=1,
    ):

        tuning_start_time = time.perf_counter()

        (
            actual_frequency_hz,
            actual_gain_db,
        ) = configure_usrp(
            usrp,
            float(
                requested_frequency_hz
            ),
        )

        timing[
            "tuning_configuration_s"
        ] += (
            time.perf_counter()
            - tuning_start_time
        )

        measurements = []

        for _ in range(
            REPEATS_PER_FREQUENCY
        ):

            acquisition_start_time = (
                time.perf_counter()
            )

            samples = acquire_time_domain(
                usrp
            )

            timing[
                "acquisition_s"
            ] += (
                time.perf_counter()
                - acquisition_start_time
            )

            processing_start_time = (
                time.perf_counter()
            )

            measurements.append(
                temporal_band_metric(
                    samples,
                    lowpass_sos,
                )
            )

            timing[
                "processing_s"
            ] += (
                time.perf_counter()
                - processing_start_time
            )

        processing_start_time = (
            time.perf_counter()
        )

        max_dbfs_values = np.asarray(
            [
                measurement[
                    "max_dbfs"
                ]
                for measurement
                in measurements
            ],
            dtype=float,
        )

        robust_dbfs_values = np.asarray(
            [
                measurement[
                    "robust_dbfs"
                ]
                for measurement
                in measurements
            ],
            dtype=float,
        )

        median_dbfs_values = np.asarray(
            [
                measurement[
                    "median_dbfs"
                ]
                for measurement
                in measurements
            ],
            dtype=float,
        )

        metrics = {
            "max_dbfs":
                float(
                    np.median(
                        max_dbfs_values
                    )
                ),

            "robust_dbfs":
                float(
                    np.median(
                        robust_dbfs_values
                    )
                ),

            "median_dbfs":
                float(
                    np.median(
                        median_dbfs_values
                    )
                ),

            "max_dbfs_min_repeat":
                float(
                    np.min(
                        max_dbfs_values
                    )
                ),

            "max_dbfs_max_repeat":
                float(
                    np.max(
                        max_dbfs_values
                    )
                ),

            "max_dbfs_std_repeat":
                float(
                    np.std(
                        max_dbfs_values
                    )
                ),

            "peak_amplitude":
                float(
                    np.max(
                        [
                            measurement[
                                "max_amplitude"
                            ]
                            for measurement
                            in measurements
                        ]
                    )
                ),

            "clipping_fraction":
                float(
                    np.max(
                        [
                            measurement[
                                "clipping_fraction"
                            ]
                            for measurement
                            in measurements
                        ]
                    )
                ),
        }

        timing[
            "processing_s"
        ] += (
            time.perf_counter()
            - processing_start_time
        )

        calibration_start_time = (
            time.perf_counter()
        )

        (
            max_estimated_dbm,
            reference_dbm,
        ) = dbfs_to_dbm(
            gain_calibration,
            metrics[
                "max_dbfs"
            ],
            actual_frequency_hz,
        )

        (
            robust_estimated_dbm,
            _,
        ) = dbfs_to_dbm(
            gain_calibration,
            metrics[
                "robust_dbfs"
            ],
            actual_frequency_hz,
        )

        (
            median_estimated_dbm,
            _,
        ) = dbfs_to_dbm(
            gain_calibration,
            metrics[
                "median_dbfs"
            ],
            actual_frequency_hz,
        )

        timing[
            "calibration_s"
        ] += (
            time.perf_counter()
            - calibration_start_time
        )

        result = FrequencyResult(

            requested_frequency_hz=float(
                requested_frequency_hz
            ),

            actual_frequency_hz=float(
                actual_frequency_hz
            ),

            frequency_mhz=float(
                actual_frequency_hz
                / 1e6
            ),

            gain_db=float(
                actual_gain_db
            ),

            max_dbfs=float(
                metrics[
                    "max_dbfs"
                ]
            ),

            robust_dbfs=float(
                metrics[
                    "robust_dbfs"
                ]
            ),

            median_dbfs=float(
                metrics[
                    "median_dbfs"
                ]
            ),

            calibration_reference_dbm=float(
                reference_dbm
            ),

            max_estimated_dbm=float(
                max_estimated_dbm
            ),

            robust_estimated_dbm=float(
                robust_estimated_dbm
            ),

            median_estimated_dbm=float(
                median_estimated_dbm
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
            f"Gain={result.gain_db:5.1f} dB | "
            f"Ref={result.calibration_reference_dbm:7.2f} dBm | "
            f"Max={result.max_dbfs:7.2f} dBFS | "
            f"Max≈{result.max_estimated_dbm:7.2f} dBm | "
            f"Robust≈{result.robust_estimated_dbm:7.2f} dBm | "
            f"Clip={result.clipping_fraction:.2e}"
        )

    return results, scan_start_time, timing


# ============================================================
# 10) AFFICHAGE
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

    max_values_dbfs = np.asarray(
        [
            result.max_dbfs
            for result in results
        ],
        dtype=float,
    )

    robust_values_dbfs = np.asarray(
        [
            result.robust_dbfs
            for result in results
        ],
        dtype=float,
    )

    min_repeat_dbfs = np.asarray(
        [
            result.max_dbfs_min_repeat
            for result in results
        ],
        dtype=float,
    )

    max_repeat_dbfs = np.asarray(
        [
            result.max_dbfs_max_repeat
            for result in results
        ],
        dtype=float,
    )

    max_values_dbm = np.asarray(
        [
            result.max_estimated_dbm
            for result in results
        ],
        dtype=float,
    )

    robust_values_dbm = np.asarray(
        [
            result.robust_estimated_dbm
            for result in results
        ],
        dtype=float,
    )

    median_values_dbm = np.asarray(
        [
            result.median_estimated_dbm
            for result in results
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Graphe 1 : dBFS
    # --------------------------------------------------------

    plt.figure(
        figsize=(15, 7)
    )

    plt.fill_between(
        frequencies_mhz,
        min_repeat_dbfs,
        max_repeat_dbfs,
        alpha=0.18,
        label="Variation MAX entre acquisitions",
    )

    plt.plot(
        frequencies_mhz,
        max_values_dbfs,
        linewidth=1.7,
        label="Médiane du MAX temporel",
    )

    plt.plot(
        frequencies_mhz,
        robust_values_dbfs,
        linewidth=1.5,
        label=f"Percentile {ROBUST_PERCENTILE}%",
    )

    plt.title(
        "Scan temporel en dBFS — "
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

    # --------------------------------------------------------
    # Graphe 2 : dBm avec TA calibration
    # --------------------------------------------------------

    plt.figure(
        figsize=(15, 7)
    )

    plt.plot(
        frequencies_mhz,
        max_values_dbm,
        linewidth=1.8,
        label="MAX temporel calibré",
    )

    plt.plot(
        frequencies_mhz,
        robust_values_dbm,
        linewidth=1.5,
        label=(
            f"Percentile "
            f"{ROBUST_PERCENTILE}% calibré"
        ),
    )

    # plt.plot(
    #     frequencies_mhz,
    #     median_values_dbm,
    #     linewidth=1.2,
    #     label="Médiane temporelle calibrée",
    # )

    plt.title(
        "Spectre temporel calibré avec "
        "rx2_power_calibration_final.pickle — "
        f"gain RX = {GAIN_DB:.0f} dB"
    )

    plt.xlabel(
        "Fréquence centrale RX (MHz)"
    )

    plt.ylabel(
        "Puissance estimée à l'entrée RX2 (dBm)"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 11) MAIN
# ============================================================

def main() -> None:

    print(
        "Chargement de TA calibration UHD..."
    )

    calibration_table = (
        load_power_calibration(
            CAL_FILE
        )
    )

    gain_calibration = (
        build_gain_calibration(
            calibration_table,
            GAIN_DB,
        )
    )

    selected_gain = float(
        gain_calibration[
            "gain_db"
        ]
    )

    minimum_frequency_mhz = float(
        gain_calibration[
            "frequencies_hz"
        ].min()
        / 1e6
    )

    maximum_frequency_mhz = float(
        gain_calibration[
            "frequencies_hz"
        ].max()
        / 1e6
    )

    print(
        f"Calibration utilisée : "
        f"gain {selected_gain:.1f} dB"
    )

    print(
        f"Plage calibrée : "
        f"{minimum_frequency_mhz:.1f} "
        f"à "
        f"{maximum_frequency_mhz:.1f} MHz"
    )

    print(
        f"Nombre de points de calibration : "
        f"{len(gain_calibration['frequencies_hz'])}"
    )

    # Vérification que le scan demandé est contenu
    # dans la table de calibration.
    if (
        F_START_HZ
        < gain_calibration[
            "frequencies_hz"
        ].min()
        - CAL_FREQUENCY_TOLERANCE_HZ
    ):
        raise ValueError(
            "F_START_HZ est hors de la plage calibrée."
        )

    if (
        F_STOP_HZ
        > gain_calibration[
            "frequencies_hz"
        ].max()
        + CAL_FREQUENCY_TOLERANCE_HZ
    ):
        raise ValueError(
            "F_STOP_HZ est hors de la plage calibrée."
        )

    (
        results,
        scan_start_time,
        timing,
    ) = scan_pd_time_domain(
        gain_calibration
    )

    display_start_time = (
        time.perf_counter()
    )

    plot_temporal_scan(
        results
    )

    display_time = (
        time.perf_counter()
        - display_start_time
    )

    elapsed_time = (
        time.perf_counter()
        - scan_start_time
    )

    measured_steps_time = (
        timing[
            "tuning_configuration_s"
        ]
        + timing[
            "acquisition_s"
        ]
        + timing[
            "processing_s"
        ]
        + timing[
            "calibration_s"
        ]
        + display_time
    )

    other_time = (
        elapsed_time
        - measured_steps_time
    )

    print()
    print(
        "============================================================"
    )
    print(
        "DÉTAIL DES TEMPS"
    )
    print(
        "============================================================"
    )
    print(
        f"Tuning / configuration USRP : "
        f"{timing['tuning_configuration_s']:.3f} s"
    )
    print(
        f"Acquisition I/Q             : "
        f"{timing['acquisition_s']:.3f} s"
    )
    print(
        f"Traitement signal           : "
        f"{timing['processing_s']:.3f} s"
    )
    print(
        f"Calibration dBFS -> dBm     : "
        f"{timing['calibration_s']:.3f} s"
    )
    print(
        f"Affichage du spectre        : "
        f"{display_time:.3f} s"
    )
    print(
        f"Autres opérations Python    : "
        f"{other_time:.3f} s"
    )
    print(
        "------------------------------------------------------------"
    )
    print(
        f"TEMPS TOTAL                  : "
        f"{elapsed_time:.3f} s"
    )
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()