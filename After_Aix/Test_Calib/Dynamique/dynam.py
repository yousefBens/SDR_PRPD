#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvisa
import uhd


# ============================================================
# 1) CONFIGURATION DU BANC
# ============================================================

USRP_SERIAL = "306BD15"
CHANNEL = 0
ANTENNA = "RX2"

GENERATOR_IP = "192.168.1.100"

GENERATOR_MIN_DBM = -20.0
GENERATOR_MAX_DBM = 20.0

# Limites de sécurité à l'entrée RX2.
RX2_DANGEROUS_DBM = -15.0
RX2_SAFE_MAX_DBM = -20.0

RATE_HZ = 12e6
ACQUISITION_DURATION_S = 0.020
DISCARD_DURATION_S = 0.002
SETTLING_TIME_S = 0.15

# Décalage matériel du LO pour éviter les défauts autour de DC.
LO_OFFSET_HZ = 2e6

# Le générateur est placé à fréquence_centrale + 1 MHz.
# La puissance du signal est mesurée précisément dans cette raie.
TEST_TONE_OFFSET_HZ = 1e6

FREQUENCIES_HZ = np.arange(
    100e6,
    2e9 + 100e6,
    100e6,
)

# Tous les gains sont mesurés avec tous les atténuateurs.
# Cela permet d'obtenir, pour chaque gain, une plage RX2 de -80 à -20 dBm.
GAINS_DB = [0.0, 20.0, 40.0, 60.0, 76.0]
ATTENUATIONS_DB = [60.0, 40.0, 20.0]

GENERATOR_STEP_DB = 1.0

MEASUREMENTS_PER_POINT = 7
NOISE_MEASUREMENTS = 15

# Niveaux affichés dans les graphes "puissance mesurée selon la fréquence".
PLOT_INPUT_LEVELS_DBM = [
    -80.0,
    -70.0,
    -60.0,
    -50.0,
    -40.0,
    -30.0,
    -20.0,
]

OUTPUT_DIR = Path.home() / "b200_dynamic_range_complete"

RAW_CSV = OUTPUT_DIR / "dynamic_range_raw_measurements.csv"
POINTS_CSV = OUTPUT_DIR / "dynamic_range_aggregated_points.csv"
SUMMARY_CSV = OUTPUT_DIR / "dynamic_range_summary.csv"
PLOTS_DIR = OUTPUT_DIR / "plots"


# ============================================================
# 2) CRITÈRES DE DÉTECTION ET DE LINÉARITÉ
# ============================================================

MIN_SNR_DB = 6.0
MIN_DETECTION_PROBABILITY = 0.90

MAX_TOTAL_POWER_DBFS = -3.0
CLIPPING_AMPLITUDE = 0.98
MAX_CLIPPING_FRACTION = 1e-4

MAX_COMPRESSION_DB = 1.0

LINEAR_FIT_MIN_SNR_DB = 12.0
LINEAR_FIT_MAX_DBFS = -12.0

EPSILON = 1e-20


# ============================================================
# 3) STRUCTURES DE DONNÉES
# ============================================================

@dataclass
class RawMeasurement:
    frequency_hz: float
    frequency_mhz: float
    generator_frequency_hz: float
    tone_offset_hz: float

    gain_db: float
    attenuation_db: float

    generator_dbm: float
    input_rx2_dbm: float

    total_power_linear: float
    total_power_dbfs: float

    tone_power_linear: float
    tone_power_dbfs: float

    noise_total_power_linear: float
    noise_total_dbfs: float

    noise_tone_power_linear: float
    noise_tone_dbfs: float

    snr_db: float
    peak_amplitude: float
    clipping_fraction: float
    detected: bool


@dataclass
class AggregatedPoint:
    frequency_hz: float
    frequency_mhz: float
    gain_db: float
    input_rx2_dbm: float

    generator_dbm: float
    attenuation_db: float

    total_power_dbfs: float
    tone_power_dbfs: float

    noise_total_dbfs: float
    noise_tone_dbfs: float

    median_snr_db: float
    detection_probability: float

    peak_amplitude: float
    clipping_fraction: float

    predicted_dbfs: float = math.nan
    compression_db: float = math.nan
    linear: bool = False


# ============================================================
# 4) GÉNÉRATEUR N5183A
# ============================================================

class N5183A:
    def __init__(self, ip_address: str):
        self.rm = pyvisa.ResourceManager("@py")
        self.dev = self.rm.open_resource(
            f"TCPIP0::{ip_address}::inst0::INSTR"
        )

        self.dev.timeout = 10000
        self.dev.read_termination = "\n"
        self.dev.write_termination = "\n"

        identity = self.dev.query("*IDN?").strip()

        if "N5183A" not in identity.upper():
            raise RuntimeError(
                f"Instrument inattendu : {identity}"
            )

        print(f"[N5183A] {identity}")

        self.output(False)
        self.dev.write(":FREQ:MODE FIX")
        self.dev.write(":POW:MODE FIX")
        self.check_errors("initialisation")

    def set_frequency(self, frequency_hz: float) -> float:
        self.dev.write(
            f":FREQ {float(frequency_hz):.3f} HZ"
        )

        actual = float(self.dev.query(":FREQ?"))
        self.check_errors("fréquence")
        return actual

    def set_power(self, power_dbm: float) -> float:
        power_dbm = float(power_dbm)

        if not GENERATOR_MIN_DBM <= power_dbm <= GENERATOR_MAX_DBM:
            raise ValueError(
                f"Puissance générateur interdite : {power_dbm:.2f} dBm. "
                f"Plage autorisée : {GENERATOR_MIN_DBM:.1f} à "
                f"{GENERATOR_MAX_DBM:.1f} dBm."
            )

        self.dev.write(f":POW {power_dbm:.3f} DBM")
        self.check_errors("puissance")

        actual = float(self.dev.query(":POW?"))

        if abs(actual - power_dbm) > 0.1:
            self.output(False)
            raise RuntimeError(
                f"Le générateur a configuré {actual:.2f} dBm "
                f"au lieu de {power_dbm:.2f} dBm."
            )

        return actual

    def output(self, enabled: bool) -> None:
        self.dev.write(
            ":OUTP ON" if enabled else ":OUTP OFF"
        )

        state = int(float(self.dev.query(":OUTP?")))

        if state != int(enabled):
            raise RuntimeError(
                "Échec de modification de la sortie RF."
            )

        self.check_errors("RF Output")

    def check_errors(self, operation: str) -> None:
        errors = []

        while True:
            response = self.dev.query("SYST:ERR?").strip()

            if response.startswith("+0") or response.startswith("0"):
                break

            errors.append(response)

        if errors:
            raise RuntimeError(
                f"Erreur SCPI ({operation}) : "
                + " | ".join(errors)
            )

    def close(self) -> None:
        try:
            self.output(False)
        finally:
            self.dev.close()
            self.rm.close()


# ============================================================
# 5) CONFIGURATION ET ACQUISITION USRP
# ============================================================

def configure_usrp(
    usrp,
    center_frequency_hz: float,
    requested_gain_db: float,
) -> float:
    usrp.set_rx_rate(RATE_HZ, CHANNEL)
    usrp.set_rx_antenna(ANTENNA, CHANNEL)
    usrp.set_rx_gain(requested_gain_db, CHANNEL)

    tune_request = uhd.types.TuneRequest(
        center_frequency_hz,
        LO_OFFSET_HZ,
    )

    usrp.set_rx_freq(tune_request, CHANNEL)
    time.sleep(SETTLING_TIME_S)

    actual_gain = float(usrp.get_rx_gain(CHANNEL))
    actual_frequency = float(usrp.get_rx_freq(CHANNEL))

    print(
        f"USRP : fc={actual_frequency / 1e6:.3f} MHz, "
        f"gain={actual_gain:.1f} dB"
    )

    return actual_gain


def acquire_samples(usrp) -> np.ndarray:
    number_of_samples = int(
        (
            ACQUISITION_DURATION_S
            + DISCARD_DURATION_S
        )
        * RATE_HZ
    )

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [CHANNEL]

    streamer = usrp.get_rx_stream(stream_args)
    metadata = uhd.types.RXMetadata()

    samples = np.zeros(
        number_of_samples,
        dtype=np.complex64,
    )

    buffer = np.zeros(
        (1, streamer.get_max_num_samps()),
        dtype=np.complex64,
    )

    stream_command = uhd.types.StreamCMD(
        uhd.types.StreamMode.num_done
    )

    stream_command.num_samps = number_of_samples
    stream_command.stream_now = True
    streamer.issue_stream_cmd(stream_command)

    total_received = 0

    while total_received < number_of_samples:
        received = streamer.recv(
            buffer,
            metadata,
            3.0,
        )

        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            raise RuntimeError(metadata.strerror())

        if received <= 0:
            raise RuntimeError("Aucun échantillon reçu.")

        count = min(
            received,
            number_of_samples - total_received,
        )

        samples[
            total_received:total_received + count
        ] = buffer[0, :count]

        total_received += count

    discarded_samples = int(
        DISCARD_DURATION_S * RATE_HZ
    )

    return samples[discarded_samples:]


# ============================================================
# 6) CALCULS DE PUISSANCE
# ============================================================

def power_to_dbfs(power_linear: float) -> float:
    return float(
        10.0 * math.log10(
            max(power_linear, EPSILON)
        )
    )


def estimate_tone_power(
    samples: np.ndarray,
    tone_offset_hz: float,
) -> float:
    """
    Mesure cohérente de la puissance de la raie injectée.

    Le générateur est à fc + tone_offset_hz.
    Après réception, on translate cette raie à DC puis on calcule
    la puissance du coefficient complexe moyen.

    Cette métrique est beaucoup plus adaptée à un CW que la puissance
    moyenne large bande, car elle ne mélange pas tout le bruit des 12 MHz.
    """
    n = np.arange(samples.size, dtype=np.float64)

    oscillator = np.exp(
        -1j * 2.0 * np.pi * tone_offset_hz * n / RATE_HZ
    )

    complex_amplitude = np.mean(
        samples.astype(np.complex128) * oscillator
    )

    return float(abs(complex_amplitude) ** 2)


def calculate_metrics(
    samples: np.ndarray,
) -> tuple[float, float, float, float]:
    magnitude = np.abs(samples)

    total_power_linear = float(
        np.mean(magnitude**2)
    )

    tone_power_linear = estimate_tone_power(
        samples,
        TEST_TONE_OFFSET_HZ,
    )

    peak_amplitude = float(
        np.max(magnitude)
    )

    clipping_fraction = float(
        np.mean(
            magnitude >= CLIPPING_AMPLITUDE
        )
    )

    return (
        total_power_linear,
        tone_power_linear,
        peak_amplitude,
        clipping_fraction,
    )


def calculate_tone_snr(
    tone_power_linear: float,
    noise_tone_power_linear: float,
) -> float:
    signal_only_power = max(
        tone_power_linear - noise_tone_power_linear,
        EPSILON,
    )

    snr_linear = (
        signal_only_power
        / max(noise_tone_power_linear, EPSILON)
    )

    return float(
        10.0 * math.log10(
            max(snr_linear, EPSILON)
        )
    )


def measure_noise(
    usrp,
) -> tuple[float, float, float, float]:
    total_powers = []
    tone_bin_powers = []

    for _ in range(NOISE_MEASUREMENTS):
        samples = acquire_samples(usrp)

        (
            total_power_linear,
            tone_power_linear,
            _,
            _,
        ) = calculate_metrics(samples)

        total_powers.append(total_power_linear)
        tone_bin_powers.append(tone_power_linear)

    noise_total_power_linear = float(
        np.median(total_powers)
    )

    noise_tone_power_linear = float(
        np.median(tone_bin_powers)
    )

    return (
        noise_total_power_linear,
        power_to_dbfs(noise_total_power_linear),
        noise_tone_power_linear,
        power_to_dbfs(noise_tone_power_linear),
    )


# ============================================================
# 7) SÉCURITÉ DU MONTAGE
# ============================================================

def calculate_rx2_power(
    generator_dbm: float,
    attenuation_db: float,
) -> float:
    return float(generator_dbm - attenuation_db)


def assert_safe_power(
    generator_dbm: float,
    attenuation_db: float,
) -> float:
    input_rx2_dbm = calculate_rx2_power(
        generator_dbm,
        attenuation_db,
    )

    if input_rx2_dbm > RX2_SAFE_MAX_DBM:
        raise RuntimeError(
            "\nARRÊT DE SÉCURITÉ :\n"
            f"Puissance générateur : {generator_dbm:.2f} dBm\n"
            f"Atténuation : {attenuation_db:.2f} dB\n"
            f"Puissance calculée à RX2 : {input_rx2_dbm:.2f} dBm\n"
            f"Limite logicielle sûre : {RX2_SAFE_MAX_DBM:.2f} dBm\n"
            f"Seuil considéré dangereux : {RX2_DANGEROUS_DBM:.2f} dBm"
        )

    return input_rx2_dbm


def generator_power_grid(
    attenuation_db: float,
) -> np.ndarray:
    safe_generator_max = min(
        GENERATOR_MAX_DBM,
        RX2_SAFE_MAX_DBM + attenuation_db,
    )

    if safe_generator_max < GENERATOR_MIN_DBM:
        return np.asarray([], dtype=float)

    return np.arange(
        GENERATOR_MIN_DBM,
        safe_generator_max + GENERATOR_STEP_DB / 2,
        GENERATOR_STEP_DB,
    )


def request_attenuator_confirmation(
    generator: N5183A,
    attenuation_db: float,
) -> None:
    generator.output(False)

    grid = generator_power_grid(attenuation_db)

    if grid.size == 0:
        raise RuntimeError(
            f"Aucune puissance sûre avec {attenuation_db:.1f} dB."
        )

    print("\n" + "#" * 76)
    print(
        f"INSTALLE MAINTENANT {attenuation_db:.1f} dB "
        "D'ATTÉNUATION"
    )
    print(
        "Montage : N5183A -> atténuateur(s) -> RX2 du B200"
    )
    print(
        "Tous les gains seront mesurés avec ce montage : "
        f"{GAINS_DB}"
    )
    print(
        f"Plage générateur : {grid[0]:.1f} à {grid[-1]:.1f} dBm"
    )
    print(
        f"Plage RX2 : "
        f"{grid[0] - attenuation_db:.1f} à "
        f"{grid[-1] - attenuation_db:.1f} dBm"
    )
    print(
        f"Le script interdit RX2 > {RX2_SAFE_MAX_DBM:.1f} dBm."
    )
    print("La sortie RF est OFF.")

    answer = input(
        "Tape exactement OUI après vérification du montage : "
    ).strip().upper()

    if answer != "OUI":
        raise RuntimeError(
            "Mesure annulée : montage non confirmé."
        )


# ============================================================
# 8) MESURE D'UN POINT
# ============================================================

def measure_generator_point(
    usrp,
    generator: N5183A,
    center_frequency_hz: float,
    generator_frequency_hz: float,
    gain_db: float,
    attenuation_db: float,
    requested_generator_dbm: float,
    noise_total_power_linear: float,
    noise_total_dbfs: float,
    noise_tone_power_linear: float,
    noise_tone_dbfs: float,
) -> list[RawMeasurement]:
    assert_safe_power(
        requested_generator_dbm,
        attenuation_db,
    )

    actual_generator_dbm = generator.set_power(
        requested_generator_dbm
    )

    actual_input_rx2_dbm = assert_safe_power(
        actual_generator_dbm,
        attenuation_db,
    )

    time.sleep(SETTLING_TIME_S)

    measurements = []

    for _ in range(MEASUREMENTS_PER_POINT):
        samples = acquire_samples(usrp)

        (
            total_power_linear,
            tone_power_linear,
            peak_amplitude,
            clipping_fraction,
        ) = calculate_metrics(samples)

        total_power_dbfs = power_to_dbfs(
            total_power_linear
        )

        tone_power_dbfs = power_to_dbfs(
            tone_power_linear
        )

        snr_db = calculate_tone_snr(
            tone_power_linear,
            noise_tone_power_linear,
        )

        measurements.append(
            RawMeasurement(
                frequency_hz=float(center_frequency_hz),
                frequency_mhz=float(
                    center_frequency_hz / 1e6
                ),
                generator_frequency_hz=float(
                    generator_frequency_hz
                ),
                tone_offset_hz=float(
                    TEST_TONE_OFFSET_HZ
                ),
                gain_db=float(gain_db),
                attenuation_db=float(
                    attenuation_db
                ),
                generator_dbm=float(
                    actual_generator_dbm
                ),
                input_rx2_dbm=float(
                    actual_input_rx2_dbm
                ),
                total_power_linear=float(
                    total_power_linear
                ),
                total_power_dbfs=float(
                    total_power_dbfs
                ),
                tone_power_linear=float(
                    tone_power_linear
                ),
                tone_power_dbfs=float(
                    tone_power_dbfs
                ),
                noise_total_power_linear=float(
                    noise_total_power_linear
                ),
                noise_total_dbfs=float(
                    noise_total_dbfs
                ),
                noise_tone_power_linear=float(
                    noise_tone_power_linear
                ),
                noise_tone_dbfs=float(
                    noise_tone_dbfs
                ),
                snr_db=float(snr_db),
                peak_amplitude=float(
                    peak_amplitude
                ),
                clipping_fraction=float(
                    clipping_fraction
                ),
                detected=bool(
                    snr_db >= MIN_SNR_DB
                ),
            )
        )

    return measurements


# ============================================================
# 9) AGRÉGATION
# ============================================================

def aggregate_all_measurements(
    measurements: list[RawMeasurement],
) -> list[AggregatedPoint]:
    """
    Fusionne les mesures issues des différents atténuateurs.

    Une même puissance RX2 peut être obtenue avec plusieurs montages.
    Ces points sont fusionnés par médiane.
    """
    grouped: dict[
        tuple[float, float, float],
        list[RawMeasurement]
    ] = {}

    for measurement in measurements:
        key = (
            round(measurement.frequency_hz, 1),
            round(measurement.gain_db, 3),
            round(measurement.input_rx2_dbm, 3),
        )

        grouped.setdefault(key, []).append(measurement)

    points = []

    for (
        frequency_hz,
        gain_db,
        input_rx2_dbm,
    ), group in grouped.items():
        points.append(
            AggregatedPoint(
                frequency_hz=float(frequency_hz),
                frequency_mhz=float(
                    frequency_hz / 1e6
                ),
                gain_db=float(gain_db),
                input_rx2_dbm=float(
                    input_rx2_dbm
                ),
                generator_dbm=float(
                    np.median(
                        [m.generator_dbm for m in group]
                    )
                ),
                attenuation_db=float(
                    np.median(
                        [m.attenuation_db for m in group]
                    )
                ),
                total_power_dbfs=float(
                    np.median(
                        [m.total_power_dbfs for m in group]
                    )
                ),
                tone_power_dbfs=float(
                    np.median(
                        [m.tone_power_dbfs for m in group]
                    )
                ),
                noise_total_dbfs=float(
                    np.median(
                        [m.noise_total_dbfs for m in group]
                    )
                ),
                noise_tone_dbfs=float(
                    np.median(
                        [m.noise_tone_dbfs for m in group]
                    )
                ),
                median_snr_db=float(
                    np.median(
                        [m.snr_db for m in group]
                    )
                ),
                detection_probability=float(
                    np.mean(
                        [m.detected for m in group]
                    )
                ),
                peak_amplitude=float(
                    np.max(
                        [m.peak_amplitude for m in group]
                    )
                ),
                clipping_fraction=float(
                    np.max(
                        [m.clipping_fraction for m in group]
                    )
                ),
            )
        )

    return sorted(
        points,
        key=lambda point: (
            point.gain_db,
            point.frequency_hz,
            point.input_rx2_dbm,
        ),
    )


def fit_linear_region(
    points: list[AggregatedPoint],
) -> Optional[tuple[float, float]]:
    fit_points = [
        point
        for point in points
        if (
            point.median_snr_db
            >= LINEAR_FIT_MIN_SNR_DB
            and point.tone_power_dbfs
            <= LINEAR_FIT_MAX_DBFS
            and point.total_power_dbfs
            <= MAX_TOTAL_POWER_DBFS
            and point.clipping_fraction
            <= MAX_CLIPPING_FRACTION
        )
    ]

    if len(fit_points) < 3:
        return None

    x = np.asarray(
        [point.input_rx2_dbm for point in fit_points],
        dtype=float,
    )

    y = np.asarray(
        [point.tone_power_dbfs for point in fit_points],
        dtype=float,
    )

    slope, intercept = np.polyfit(x, y, 1)

    return float(slope), float(intercept)


def evaluate_linearity(
    points: list[AggregatedPoint],
    fit_parameters: Optional[tuple[float, float]],
) -> None:
    if fit_parameters is None:
        return

    slope, intercept = fit_parameters

    for point in points:
        point.predicted_dbfs = (
            slope * point.input_rx2_dbm
            + intercept
        )

        point.compression_db = (
            point.predicted_dbfs
            - point.tone_power_dbfs
        )

        point.linear = bool(
            point.detection_probability
            >= MIN_DETECTION_PROBABILITY
            and point.total_power_dbfs
            <= MAX_TOTAL_POWER_DBFS
            and point.clipping_fraction
            <= MAX_CLIPPING_FRACTION
            and abs(point.compression_db)
            <= MAX_COMPRESSION_DB
        )


def estimate_pmin(
    points: list[AggregatedPoint],
) -> tuple[Optional[float], str]:
    detected_points = [
        point
        for point in points
        if (
            point.detection_probability
            >= MIN_DETECTION_PROBABILITY
        )
    ]

    if not detected_points:
        return None, "aucun_signal_detecte"

    pmin = min(
        point.input_rx2_dbm
        for point in detected_points
    )

    minimum_tested = min(
        point.input_rx2_dbm
        for point in points
    )

    if np.isclose(
        pmin,
        minimum_tested,
        atol=0.1,
    ):
        return pmin, "limite_basse_du_banc"

    return pmin, "mesure_valide"


def estimate_pmax(
    points: list[AggregatedPoint],
) -> tuple[Optional[float], str]:
    linear_points = [
        point
        for point in points
        if point.linear
    ]

    if not linear_points:
        return None, "aucun_point_lineaire"

    pmax = max(
        point.input_rx2_dbm
        for point in linear_points
    )

    maximum_tested = max(
        point.input_rx2_dbm
        for point in points
    )

    if np.isclose(
        pmax,
        maximum_tested,
        atol=0.1,
    ):
        return pmax, "limite_haute_du_banc"

    return pmax, "mesure_valide"


def build_summary(
    points: list[AggregatedPoint],
) -> list[dict]:
    summary_rows = []

    groups: dict[
        tuple[float, float],
        list[AggregatedPoint]
    ] = {}

    for point in points:
        key = (
            round(point.frequency_hz, 1),
            round(point.gain_db, 3),
        )

        groups.setdefault(key, []).append(point)

    for (
        frequency_hz,
        gain_db,
    ), case_points in sorted(groups.items()):
        case_points = sorted(
            case_points,
            key=lambda point: point.input_rx2_dbm,
        )

        fit_parameters = fit_linear_region(
            case_points
        )

        evaluate_linearity(
            case_points,
            fit_parameters,
        )

        pmin_dbm, pmin_status = estimate_pmin(
            case_points
        )

        pmax_dbm, pmax_status = estimate_pmax(
            case_points
        )

        dynamic_range_db = None

        if (
            pmin_dbm is not None
            and pmax_dbm is not None
        ):
            dynamic_range_db = (
                pmax_dbm - pmin_dbm
            )

        slope = (
            fit_parameters[0]
            if fit_parameters is not None
            else None
        )

        intercept = (
            fit_parameters[1]
            if fit_parameters is not None
            else None
        )

        summary_rows.append(
            {
                "frequency_hz": float(frequency_hz),
                "frequency_mhz": float(
                    frequency_hz / 1e6
                ),
                "gain_db": float(gain_db),
                "rx2_min_tested_dbm": float(
                    min(p.input_rx2_dbm for p in case_points)
                ),
                "rx2_max_tested_dbm": float(
                    max(p.input_rx2_dbm for p in case_points)
                ),
                "noise_total_dbfs": float(
                    np.median(
                        [p.noise_total_dbfs for p in case_points]
                    )
                ),
                "noise_tone_dbfs": float(
                    np.median(
                        [p.noise_tone_dbfs for p in case_points]
                    )
                ),
                "minimum_detectable_dbm": (
                    ""
                    if pmin_dbm is None
                    else float(pmin_dbm)
                ),
                "pmin_status": pmin_status,
                "maximum_linear_dbm": (
                    ""
                    if pmax_dbm is None
                    else float(pmax_dbm)
                ),
                "pmax_status": pmax_status,
                "dynamic_range_db": (
                    ""
                    if dynamic_range_db is None
                    else float(dynamic_range_db)
                ),
                "linear_fit_slope": (
                    ""
                    if slope is None
                    else float(slope)
                ),
                "linear_fit_intercept": (
                    ""
                    if intercept is None
                    else float(intercept)
                ),
                "number_of_input_points": len(
                    case_points
                ),
            }
        )

    return summary_rows


# ============================================================
# 10) SAUVEGARDE CSV
# ============================================================

def save_dataclass_rows(
    path: Path,
    rows: list,
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                asdict(rows[0]).keys()
            ),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def save_dict_rows(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 11) GRAPHES
# ============================================================

def save_all_plots(
    points_csv: Path,
    summary_csv: Path,
) -> None:
    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    points_df = pd.read_csv(points_csv)
    summary_df = pd.read_csv(summary_csv)

    points_df = points_df.sort_values(
        ["gain_db", "frequency_mhz", "input_rx2_dbm"]
    )

    summary_df = summary_df.sort_values(
        ["gain_db", "frequency_mhz"]
    )

    gains = sorted(
        points_df["gain_db"].dropna().unique()
    )

    # --------------------------------------------------------
    # Graphe 1 :
    # pour chaque gain, bruit + puissances captées à différents Pin.
    # --------------------------------------------------------
    for gain_db in gains:
        gain_points = points_df[
            np.isclose(points_df["gain_db"], gain_db)
        ].copy()

        gain_summary = summary_df[
            np.isclose(summary_df["gain_db"], gain_db)
        ].copy()

        plt.figure(figsize=(13, 7))

        plt.plot(
            gain_summary["frequency_mhz"],
            gain_summary["noise_tone_dbfs"],
            linestyle="--",
            linewidth=2.5,
            label="Noise floor dans la raie de mesure",
        )

        for input_level_dbm in PLOT_INPUT_LEVELS_DBM:
            level_data = gain_points[
                np.isclose(
                    gain_points["input_rx2_dbm"],
                    input_level_dbm,
                    atol=0.1,
                )
            ].sort_values("frequency_mhz")

            if level_data.empty:
                continue

            plt.plot(
                level_data["frequency_mhz"],
                level_data["tone_power_dbfs"],
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=f"Entrée RX2 = {input_level_dbm:.0f} dBm",
            )

        plt.title(
            f"B200 — bruit et puissance CW captée selon la fréquence "
            f"(gain RX = {gain_db:.0f} dB)"
        )
        plt.xlabel("Fréquence centrale (MHz)")
        plt.ylabel("Puissance mesurée (dBFS)")
        plt.grid(True, alpha=0.3)
        plt.legend(
            loc="best",
            ncol=2,
            fontsize=9,
        )
        plt.tight_layout()

        output = (
            PLOTS_DIR
            / f"gain_{gain_db:.0f}_noise_and_measured_power.png"
        )

        plt.savefig(output, dpi=180)
        plt.close()

    # --------------------------------------------------------
    # Graphe 2 :
    # plage détectable et linéaire selon la fréquence.
    # --------------------------------------------------------
    for gain_db in gains:
        gain_summary = summary_df[
            np.isclose(summary_df["gain_db"], gain_db)
        ].copy()

        gain_summary["minimum_detectable_dbm"] = pd.to_numeric(
            gain_summary["minimum_detectable_dbm"],
            errors="coerce",
        )

        gain_summary["maximum_linear_dbm"] = pd.to_numeric(
            gain_summary["maximum_linear_dbm"],
            errors="coerce",
        )

        valid = gain_summary.dropna(
            subset=[
                "minimum_detectable_dbm",
                "maximum_linear_dbm",
            ]
        )

        plt.figure(figsize=(13, 6))

        plt.plot(
            gain_summary["frequency_mhz"],
            gain_summary["minimum_detectable_dbm"],
            marker="o",
            label="Puissance minimale détectable",
        )

        plt.plot(
            gain_summary["frequency_mhz"],
            gain_summary["maximum_linear_dbm"],
            marker="o",
            label="Puissance maximale linéaire",
        )

        if not valid.empty:
            plt.fill_between(
                valid["frequency_mhz"],
                valid["minimum_detectable_dbm"],
                valid["maximum_linear_dbm"],
                alpha=0.2,
                label="Zone dynamique exploitable",
            )

        plt.title(
            f"Plage dynamique d'entrée selon la fréquence "
            f"(gain RX = {gain_db:.0f} dB)"
        )
        plt.xlabel("Fréquence centrale (MHz)")
        plt.ylabel("Puissance à l'entrée RX2 (dBm)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        output = (
            PLOTS_DIR
            / f"gain_{gain_db:.0f}_input_dynamic_limits.png"
        )

        plt.savefig(output, dpi=180)
        plt.close()

    # --------------------------------------------------------
    # Graphe 3 :
    # dynamique en dB selon la fréquence, tous les gains.
    # --------------------------------------------------------
    plt.figure(figsize=(13, 7))

    for gain_db in gains:
        gain_summary = summary_df[
            np.isclose(summary_df["gain_db"], gain_db)
        ].copy()

        gain_summary["dynamic_range_db"] = pd.to_numeric(
            gain_summary["dynamic_range_db"],
            errors="coerce",
        )

        gain_summary = gain_summary.dropna(
            subset=["dynamic_range_db"]
        )

        if gain_summary.empty:
            continue

        plt.plot(
            gain_summary["frequency_mhz"],
            gain_summary["dynamic_range_db"],
            marker="o",
            label=f"Gain {gain_db:.0f} dB",
        )

    plt.title(
        "Dynamique mesurée du B200 selon la fréquence et le gain"
    )
    plt.xlabel("Fréquence centrale (MHz)")
    plt.ylabel("Dynamique d'entrée (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "dynamic_range_all_gains.png",
        dpi=180,
    )
    plt.close()

    # --------------------------------------------------------
    # Graphe 4 :
    # noise floor total selon fréquence pour tous les gains.
    # --------------------------------------------------------
    plt.figure(figsize=(13, 7))

    for gain_db in gains:
        gain_summary = summary_df[
            np.isclose(summary_df["gain_db"], gain_db)
        ].copy()

        plt.plot(
            gain_summary["frequency_mhz"],
            gain_summary["noise_total_dbfs"],
            marker="o",
            label=f"Gain {gain_db:.0f} dB",
        )

    plt.title(
        "Noise floor large bande du B200 selon la fréquence"
    )
    plt.xlabel("Fréquence centrale (MHz)")
    plt.ylabel("Bruit moyen sur 12 MHz (dBFS)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "noise_floor_all_gains.png",
        dpi=180,
    )
    plt.close()


# ============================================================
# 12) PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    usrp = uhd.usrp.MultiUSRP(
        f"serial={USRP_SERIAL}"
    )

    generator = N5183A(
        GENERATOR_IP
    )

    all_measurements: list[RawMeasurement] = []

    try:
        for attenuation_db in ATTENUATIONS_DB:
            attenuation_db = float(attenuation_db)

            request_attenuator_confirmation(
                generator,
                attenuation_db,
            )

            grid = generator_power_grid(
                attenuation_db
            )

            print(
                f"Balayage générateur : "
                f"{grid[0]:.1f} à {grid[-1]:.1f} dBm"
            )
            print(
                f"Balayage RX2 : "
                f"{grid[0] - attenuation_db:.1f} à "
                f"{grid[-1] - attenuation_db:.1f} dBm"
            )

            for center_frequency_hz in FREQUENCIES_HZ:
                center_frequency_hz = float(
                    center_frequency_hz
                )

                generator_frequency_hz = (
                    center_frequency_hz
                    + TEST_TONE_OFFSET_HZ
                )

                generator.output(False)

                actual_generator_frequency_hz = (
                    generator.set_frequency(
                        generator_frequency_hz
                    )
                )

                for requested_gain_db in GAINS_DB:
                    print("\n" + "=" * 76)
                    print(
                        f"fc={center_frequency_hz / 1e6:.1f} MHz | "
                        f"fgen={actual_generator_frequency_hz / 1e6:.1f} MHz | "
                        f"gain demandé={requested_gain_db:.1f} dB | "
                        f"atténuation={attenuation_db:.1f} dB"
                    )

                    actual_gain_db = configure_usrp(
                        usrp,
                        center_frequency_hz,
                        requested_gain_db,
                    )

                    generator.output(False)
                    time.sleep(SETTLING_TIME_S)

                    (
                        noise_total_power_linear,
                        noise_total_dbfs,
                        noise_tone_power_linear,
                        noise_tone_dbfs,
                    ) = measure_noise(usrp)

                    print(
                        f"Bruit large bande : "
                        f"{noise_total_dbfs:.2f} dBFS"
                    )

                    print(
                        f"Bruit raie {TEST_TONE_OFFSET_HZ / 1e6:.1f} MHz : "
                        f"{noise_tone_dbfs:.2f} dBFS"
                    )

                    generator.set_power(
                        float(grid[0])
                    )

                    assert_safe_power(
                        float(grid[0]),
                        attenuation_db,
                    )

                    generator.output(True)

                    try:
                        for generator_dbm in grid:
                            measurements = measure_generator_point(
                                usrp=usrp,
                                generator=generator,
                                center_frequency_hz=center_frequency_hz,
                                generator_frequency_hz=actual_generator_frequency_hz,
                                gain_db=actual_gain_db,
                                attenuation_db=attenuation_db,
                                requested_generator_dbm=float(
                                    generator_dbm
                                ),
                                noise_total_power_linear=noise_total_power_linear,
                                noise_total_dbfs=noise_total_dbfs,
                                noise_tone_power_linear=noise_tone_power_linear,
                                noise_tone_dbfs=noise_tone_dbfs,
                            )

                            all_measurements.extend(
                                measurements
                            )

                            median_tone_dbfs = float(
                                np.median(
                                    [
                                        m.tone_power_dbfs
                                        for m in measurements
                                    ]
                                )
                            )

                            median_total_dbfs = float(
                                np.median(
                                    [
                                        m.total_power_dbfs
                                        for m in measurements
                                    ]
                                )
                            )

                            median_snr = float(
                                np.median(
                                    [
                                        m.snr_db
                                        for m in measurements
                                    ]
                                )
                            )

                            detection_probability = float(
                                np.mean(
                                    [
                                        m.detected
                                        for m in measurements
                                    ]
                                )
                            )

                            max_clip = float(
                                np.max(
                                    [
                                        m.clipping_fraction
                                        for m in measurements
                                    ]
                                )
                            )

                            input_dbm = (
                                float(generator_dbm)
                                - attenuation_db
                            )

                            print(
                                f"Gen={generator_dbm:6.1f} dBm | "
                                f"RX2={input_dbm:7.1f} dBm | "
                                f"Tone={median_tone_dbfs:7.2f} dBFS | "
                                f"Total={median_total_dbfs:7.2f} dBFS | "
                                f"SNR={median_snr:6.2f} dB | "
                                f"Pd={detection_probability:.2f} | "
                                f"clip={max_clip:.2e}"
                            )

                            # Sauvegarde progressive en cas d'arrêt.
                            save_dataclass_rows(
                                RAW_CSV,
                                all_measurements,
                            )

                    finally:
                        generator.output(False)

    finally:
        generator.close()

    print("\nAgrégation des mesures...")

    aggregated_points = aggregate_all_measurements(
        all_measurements
    )

    summary_rows = build_summary(
        aggregated_points
    )

    save_dataclass_rows(
        POINTS_CSV,
        aggregated_points,
    )

    save_dict_rows(
        SUMMARY_CSV,
        summary_rows,
    )

    print("Création des graphes...")

    save_all_plots(
        POINTS_CSV,
        SUMMARY_CSV,
    )

    print("\nMesures terminées.")
    print(f"CSV brut       : {RAW_CSV}")
    print(f"CSV agrégé     : {POINTS_CSV}")
    print(f"CSV résumé     : {SUMMARY_CSV}")
    print(f"Graphes        : {PLOTS_DIR}")


if __name__ == "__main__":
    main()