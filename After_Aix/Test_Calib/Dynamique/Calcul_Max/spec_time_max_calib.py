#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uhd
from scipy.signal import butter, sosfiltfilt


# ============================================================
# 1) CONFIGURATION USRP ET SCAN
# ============================================================

USRP_SERIAL = "306BD15"
CHANNEL = 0
ANTENNA = "RX2"

# La calibration MAX commence à 100 MHz.
F_START_HZ = 100e6
F_STOP_HZ = 2e9

RATE_HZ = 12e6
GAIN_DB = 40.0

# Avec un filtre utile de ±4 MHz, un pas de 5 MHz donne
# un recouvrement suffisant entre les fenêtres.
STEP_HZ = 5e6

# Offset matériel du LO afin de limiter les défauts autour de DC.
LO_OFFSET_HZ = 1e6

DISCARD_DURATION_S = 0.002
USEFUL_DURATION_S = 0.020
ACQUISITION_DURATION_S = (
    DISCARD_DURATION_S
    + USEFUL_DURATION_S
)

SETTLING_TIME_S = 0.10

# Plusieurs acquisitions par fréquence.
REPEATS_PER_FREQUENCY = 5

LOWPASS_CUTOFF_HZ = 4e6
FILTER_ORDER = 4

ROBUST_PERCENTILE = 99.99

# Suppression des bords après sosfiltfilt pour éviter
# que les transitoires du filtre dominent le maximum.
FILTER_EDGE_DURATION_S = 0.0002

CLIPPING_AMPLITUDE = 0.98
EPSILON = 1e-12


# ============================================================
# 2) FICHIER DE CALIBRATION MAX TEMPOREL
# ============================================================

MAX_CALIBRATION_FILE = (
    Path.home()
    / "b200_dynamic_range_max_pulses"
    / "dynamic_range_max_aggregated_points.csv"
)

# Tolérance maximale entre le gain demandé et un gain calibré.
MAX_GAIN_DISTANCE_DB = 1.0


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

    max_estimated_dbm: float
    robust_estimated_dbm: float

    max_inside_calibrated_range: bool
    robust_inside_calibrated_range: bool

    max_dbfs_min_repeat: float
    max_dbfs_max_repeat: float
    max_dbfs_std_repeat: float

    peak_amplitude: float
    clipping_fraction: float


# ============================================================
# 4) OUTILS dBFS
# ============================================================

def amplitude_to_dbfs(amplitude: float) -> float:
    """
    Convertit une amplitude numérique normalisée en dBFS.

    Pour le maximum temporel :
        dBFS = 20*log10(max(abs(x)))
    """
    return float(
        20.0
        * np.log10(
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
# 5) CHARGEMENT DE LA CALIBRATION MAX
# ============================================================

def load_max_calibration(
    calibration_file: Path,
) -> pd.DataFrame:
    """
    Charge la calibration construite avec la même métrique MAX.

    Colonnes attendues :
        frequency_hz
        gain_db
        input_rx2_dbm
        max_dbfs
    """
    if not calibration_file.exists():
        raise FileNotFoundError(
            "Fichier de calibration MAX introuvable :\n"
            f"{calibration_file}"
        )

    calibration_df = pd.read_csv(
        calibration_file
    )

    required_columns = [
        "frequency_hz",
        "gain_db",
        "input_rx2_dbm",
        "max_dbfs",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in calibration_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le fichier de calibration : "
            + ", ".join(missing_columns)
        )

    for column in required_columns:
        calibration_df[column] = pd.to_numeric(
            calibration_df[column],
            errors="coerce",
        )

    calibration_df = calibration_df.dropna(
        subset=required_columns
    ).copy()

    if calibration_df.empty:
        raise ValueError(
            "Le fichier de calibration ne contient aucun point valide."
        )

    # Si la colonne linear existe, on garde de préférence
    # les points déclarés linéaires.
    if "linear" in calibration_df.columns:
        linear_mask = (
            calibration_df["linear"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        linear_df = calibration_df[
            linear_mask
        ].copy()

        # On ne remplace le tableau que si suffisamment de points
        # linéaires sont disponibles.
        if len(linear_df) >= 3:
            calibration_df = linear_df

    return calibration_df


def select_calibrated_gain(
    calibration_df: pd.DataFrame,
    requested_gain_db: float,
) -> float:
    """
    Sélectionne le gain calibré le plus proche.
    """
    gains = np.sort(
        calibration_df["gain_db"]
        .dropna()
        .unique()
        .astype(float)
    )

    if gains.size == 0:
        raise ValueError(
            "Aucun gain disponible dans la calibration."
        )

    selected_gain = float(
        gains[
            np.argmin(
                np.abs(
                    gains
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
            f"Gain demandé : {requested_gain_db:.1f} dB.\n"
            f"Gain calibré le plus proche : {selected_gain:.1f} dB.\n"
            f"Écart trop important : {distance:.1f} dB."
        )

    return selected_gain


def build_frequency_models(
    calibration_df: pd.DataFrame,
    gain_db: float,
) -> pd.DataFrame:
    """
    Pour chaque fréquence calibrée, construit la relation :

        input_rx2_dbm = slope * max_dbfs + intercept

    Les coefficients slope et intercept seront ensuite interpolés
    selon la fréquence.
    """
    selected_gain = select_calibrated_gain(
        calibration_df,
        gain_db,
    )

    gain_df = calibration_df[
        np.isclose(
            calibration_df["gain_db"],
            selected_gain,
            atol=0.1,
        )
    ].copy()

    model_rows = []

    for frequency_hz, case_df in gain_df.groupby(
        "frequency_hz"
    ):
        case_df = case_df.sort_values(
            "input_rx2_dbm"
        )

        x_dbfs = case_df[
            "max_dbfs"
        ].to_numpy(dtype=float)

        y_dbm = case_df[
            "input_rx2_dbm"
        ].to_numpy(dtype=float)

        valid = (
            np.isfinite(x_dbfs)
            & np.isfinite(y_dbm)
        )

        x_dbfs = x_dbfs[valid]
        y_dbm = y_dbm[valid]

        if x_dbfs.size < 3:
            continue

        # Éviter une régression impossible si tous les dBFS sont égaux.
        if np.ptp(x_dbfs) < 1e-6:
            continue

        slope, intercept = np.polyfit(
            x_dbfs,
            y_dbm,
            1,
        )

        predicted_dbm = (
            slope * x_dbfs
            + intercept
        )

        residuals = (
            y_dbm
            - predicted_dbm
        )

        rmse_db = float(
            np.sqrt(
                np.mean(
                    residuals**2
                )
            )
        )

        model_rows.append({
            "frequency_hz": float(
                frequency_hz
            ),
            "gain_db": float(
                selected_gain
            ),
            "slope": float(
                slope
            ),
            "intercept": float(
                intercept
            ),
            "minimum_calibrated_dbfs": float(
                np.min(x_dbfs)
            ),
            "maximum_calibrated_dbfs": float(
                np.max(x_dbfs)
            ),
            "minimum_calibrated_dbm": float(
                np.min(y_dbm)
            ),
            "maximum_calibrated_dbm": float(
                np.max(y_dbm)
            ),
            "calibration_rmse_db": rmse_db,
            "number_of_points": int(
                x_dbfs.size
            ),
        })

    models_df = pd.DataFrame(
        model_rows
    )

    if models_df.empty:
        raise ValueError(
            f"Impossible de construire la calibration MAX "
            f"pour le gain {selected_gain:.1f} dB."
        )

    models_df = models_df.sort_values(
        "frequency_hz"
    ).reset_index(drop=True)

    return models_df


def interpolate_calibration_model(
    models_df: pd.DataFrame,
    frequency_hz: float,
) -> dict:
    """
    Interpole les paramètres de calibration selon la fréquence.

    Une petite tolérance est appliquée aux limites pour gérer
    les écarts numériques de syntonisation UHD.
    """

    frequency_hz = float(frequency_hz)

    frequencies = (
        models_df["frequency_hz"]
        .to_numpy(dtype=float)
    )

    minimum_frequency_hz = float(
        np.min(frequencies)
    )

    maximum_frequency_hz = float(
        np.max(frequencies)
    )

    # Tolérance pour les petites différences de fréquence UHD.
    # 1 kHz est négligeable devant un pas de calibration de 100 MHz.
    frequency_tolerance_hz = 1e3

    if frequency_hz < minimum_frequency_hz:
        difference_hz = (
            minimum_frequency_hz
            - frequency_hz
        )

        if difference_hz <= frequency_tolerance_hz:
            frequency_hz = minimum_frequency_hz
        else:
            raise ValueError(
                f"Fréquence réelle "
                f"{frequency_hz / 1e6:.9f} MHz "
                "inférieure à la plage calibrée "
                f"[{minimum_frequency_hz / 1e6:.3f}, "
                f"{maximum_frequency_hz / 1e6:.3f}] MHz."
            )

    if frequency_hz > maximum_frequency_hz:
        difference_hz = (
            frequency_hz
            - maximum_frequency_hz
        )

        if difference_hz <= frequency_tolerance_hz:
            frequency_hz = maximum_frequency_hz
        else:
            raise ValueError(
                f"Fréquence réelle "
                f"{frequency_hz / 1e6:.9f} MHz "
                "supérieure à la plage calibrée "
                f"[{minimum_frequency_hz / 1e6:.3f}, "
                f"{maximum_frequency_hz / 1e6:.3f}] MHz."
            )

    def interpolate_column(
        column_name: str,
    ) -> float:
        return float(
            np.interp(
                frequency_hz,
                frequencies,
                models_df[
                    column_name
                ].to_numpy(dtype=float),
            )
        )

    return {
        "slope": interpolate_column(
            "slope"
        ),

        "intercept": interpolate_column(
            "intercept"
        ),

        "minimum_calibrated_dbfs": interpolate_column(
            "minimum_calibrated_dbfs"
        ),

        "maximum_calibrated_dbfs": interpolate_column(
            "maximum_calibrated_dbfs"
        ),

        "minimum_calibrated_dbm": interpolate_column(
            "minimum_calibrated_dbm"
        ),

        "maximum_calibrated_dbm": interpolate_column(
            "maximum_calibrated_dbm"
        ),

        "calibration_rmse_db": interpolate_column(
            "calibration_rmse_db"
        ),
    }

def max_dbfs_to_dbm(
    models_df: pd.DataFrame,
    measured_dbfs: float,
    frequency_hz: float,
) -> tuple[float, bool, float]:
    """
    Convertit un MAX dBFS en puissance estimée à l'entrée RX2.

    Retourne :
        estimated_dbm
        inside_calibrated_range
        calibration_rmse_db
    """
    model = interpolate_calibration_model(
        models_df,
        frequency_hz,
    )

    estimated_dbm = (
        model["slope"]
        * measured_dbfs
        + model["intercept"]
    )

    inside_calibrated_range = bool(
        model["minimum_calibrated_dbfs"]
        <= measured_dbfs
        <= model["maximum_calibrated_dbfs"]
    )

    return (
        float(estimated_dbm),
        inside_calibrated_range,
        float(
            model["calibration_rmse_db"]
        ),
    )


# ============================================================
# 6) CONFIGURATION USRP
# ============================================================

def configure_usrp(
    usrp: uhd.usrp.MultiUSRP,
    center_frequency_hz: float,
) -> tuple[float, float]:
    """
    Configure le B200 puis retourne la fréquence et le gain réels.
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
# 7) ACQUISITION
# ============================================================

def acquire_time_domain(
    usrp: uhd.usrp.MultiUSRP,
) -> np.ndarray:
    """
    Acquiert un bloc I/Q et retire le début de l'acquisition.
    """
    number_of_samples = int(
        ACQUISITION_DURATION_S
        * RATE_HZ
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

    stream_command.num_samps = (
        number_of_samples
    )

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
        ] = buffer[0, :count]

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
            "Aucun échantillon utile après suppression du début."
        )

    return samples


# ============================================================
# 8) CALCUL MAX TEMPOREL
# ============================================================

def temporal_band_metric(
    samples: np.ndarray,
    lowpass_sos: np.ndarray,
) -> dict:
    """
    Calcule le MAX temporel, le percentile robuste et la médiane.
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

    La mesure principale est la médiane des MAX.
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
            for measurement
            in measurements
        ],
        dtype=float,
    )

    robust_dbfs_values = np.asarray(
        [
            measurement["robust_dbfs"]
            for measurement
            in measurements
        ],
        dtype=float,
    )

    median_dbfs_values = np.asarray(
        [
            measurement["median_dbfs"]
            for measurement
            in measurements
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
                    for measurement
                    in measurements
                ]
            )
        ),

        "clipping_fraction": float(
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
    calibration_models: pd.DataFrame,
) -> list[FrequencyResult]:
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

        (
            max_estimated_dbm,
            max_inside_range,
            max_calibration_rmse_db,
        ) = max_dbfs_to_dbm(
            models_df=calibration_models,
            measured_dbfs=metrics[
                "max_dbfs"
            ],
            frequency_hz=actual_frequency_hz,
        )

        (
            robust_estimated_dbm,
            robust_inside_range,
            robust_calibration_rmse_db,
        ) = max_dbfs_to_dbm(
            models_df=calibration_models,
            measured_dbfs=metrics[
                "robust_dbfs"
            ],
            frequency_hz=actual_frequency_hz,
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
                metrics["max_dbfs"]
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

            max_estimated_dbm=float(
                max_estimated_dbm
            ),

            robust_estimated_dbm=float(
                robust_estimated_dbm
            ),

            max_inside_calibrated_range=bool(
                max_inside_range
            ),

            robust_inside_calibrated_range=bool(
                robust_inside_range
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

        calibration_status = (
            "CALIBRÉ"
            if result.max_inside_calibrated_range
            else "EXTRAPOLATION"
        )

        print(
            f"[{index:3d}/"
            f"{len(frequencies_hz)}] "
            f"{result.frequency_mhz:8.1f} MHz | "
            f"Max={result.max_dbfs:7.2f} dBFS | "
            f"Max≈{result.max_estimated_dbm:7.2f} dBm | "
            f"{calibration_status} | "
            f"RMSE calib≈"
            f"{max_calibration_rmse_db:.2f} dB | "
            f"Clip="
            f"{result.clipping_fraction:.2e}"
        )

    return results


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

    max_inside_range = np.asarray(
        [
            result.max_inside_calibrated_range
            for result in results
        ],
        dtype=bool,
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
        label=(
            "Variation du MAX "
            "entre acquisitions"
        ),
    )

    plt.plot(
        frequencies_mhz,
        max_values_dbfs,
        linewidth=1.7,
        label=(
            "Médiane du MAX temporel "
            "par fenêtre"
        ),
    )

    plt.plot(
        frequencies_mhz,
        robust_values_dbfs,
        linewidth=1.5,
        label=(
            f"Percentile "
            f"{ROBUST_PERCENTILE}%"
        ),
    )

    plt.title(
        "Détection de pulses par scan temporel — "
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
    # Graphe 2 : dBm après calibration MAX
    # --------------------------------------------------------

    plt.figure(
        figsize=(15, 7)
    )

    plt.plot(
        frequencies_mhz,
        max_values_dbm,
        linewidth=1.7,
        label=(
            "MAX temporel converti en dBm"
        ),
    )

    plt.plot(
        frequencies_mhz,
        robust_values_dbm,
        linewidth=1.5,
        label=(
            f"Percentile "
            f"{ROBUST_PERCENTILE}% "
            "converti en dBm"
        ),
    )

    outside_range = (
        ~max_inside_range
    )

    if np.any(
        outside_range
    ):
        plt.scatter(
            frequencies_mhz[
                outside_range
            ],
            max_values_dbm[
                outside_range
            ],
            marker="x",
            s=55,
            label=(
                "Valeurs hors plage calibrée "
                "(extrapolation)"
            ),
        )

    plt.title(
        "Puissance d'entrée RX2 estimée après calibration MAX — "
        f"gain RX = {GAIN_DB:.0f} dB"
    )

    plt.xlabel(
        "Fréquence centrale RX (MHz)"
    )

    plt.ylabel(
        "Puissance estimée à RX2 (dBm)"
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
    start_time = time.perf_counter()

    print(
        "Chargement de la calibration MAX..."
    )

    calibration_df = load_max_calibration(
        MAX_CALIBRATION_FILE
    )

    calibration_models = build_frequency_models(
        calibration_df,
        GAIN_DB,
    )

    selected_gain = float(
        calibration_models[
            "gain_db"
        ].iloc[0]
    )

    minimum_frequency_mhz = (
        calibration_models[
            "frequency_hz"
        ].min()
        / 1e6
    )

    maximum_frequency_mhz = (
        calibration_models[
            "frequency_hz"
        ].max()
        / 1e6
    )

    print(
        f"Calibration utilisée : gain "
        f"{selected_gain:.1f} dB"
    )

    print(
        "Plage calibrée : "
        f"{minimum_frequency_mhz:.1f} à "
        f"{maximum_frequency_mhz:.1f} MHz"
    )

    results = scan_pd_time_domain(
        calibration_models
    )

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