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
from scipy.signal import butter, sosfiltfilt


# ============================================================
# CONFIGURATION
# ============================================================

USRP_SERIAL = "306BD15"
CHANNEL = 0
ANTENNA = "RX2"

GENERATOR_IP = "192.168.1.100"
GENERATOR_MIN_DBM = -20.0
GENERATOR_MAX_DBM = 20.0

RX2_SAFE_MAX_DBM = -20.0
RX2_DANGEROUS_DBM = -15.0

RATE_HZ = 12e6
ACQUISITION_DURATION_S = 0.020
DISCARD_DURATION_S = 0.002
SETTLING_TIME_S = 0.15

LO_OFFSET_HZ = 2e6
TEST_TONE_OFFSET_HZ = 1e6

FREQUENCIES_HZ = np.arange(100e6, 2e9 + 100e6, 100e6)
GAINS_DB = [0.0, 20.0, 40.0, 60.0, 76.0]
ATTENUATIONS_DB = [60.0, 40.0, 20.0]
GENERATOR_STEP_DB = 1.0

MEASUREMENTS_PER_POINT = 7
NOISE_MEASUREMENTS = 15

# Même calcul que ton script de spectre temporel
LP_CUTOFF_HZ = 4e6
ROBUST_PERCENTILE = 99.99
REMOVE_DC = False

# Détection
MIN_MAX_EXCESS_DB = 6.0
MIN_ROBUST_EXCESS_DB = 3.0
MIN_DETECTION_PROBABILITY = 0.90

# Linéarité / saturation
LINEAR_FIT_MIN_EXCESS_DB = 12.0
LINEAR_FIT_MAX_DBFS = -12.0
MAX_COMPRESSION_DB = 1.0
MAX_ACCEPTED_MAX_DBFS = -0.5

CLIPPING_AMPLITUDE = 0.98
MAX_CLIPPING_FRACTION = 1e-4
EPSILON = 1e-20

PLOT_INPUT_LEVELS_DBM = [-80, -70, -60, -50, -40, -30, -20]

# Nouveau dossier : les anciennes mesures ne sont pas écrasées
OUTPUT_DIR = Path.home() / "b200_dynamic_range_max_pulses"
RAW_CSV = OUTPUT_DIR / "dynamic_range_max_raw_measurements.csv"
POINTS_CSV = OUTPUT_DIR / "dynamic_range_max_aggregated_points.csv"
SUMMARY_CSV = OUTPUT_DIR / "dynamic_range_max_summary.csv"
PLOTS_DIR = OUTPUT_DIR / "plots"


# ============================================================
# STRUCTURES
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

    max_amplitude: float
    max_dbfs: float
    robust_amplitude: float
    robust_dbfs: float
    median_amplitude: float
    median_dbfs: float

    noise_max_dbfs: float
    noise_robust_dbfs: float
    max_excess_db: float
    robust_excess_db: float

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

    max_dbfs: float
    robust_dbfs: float
    median_dbfs: float

    noise_max_dbfs: float
    noise_robust_dbfs: float

    median_max_excess_db: float
    median_robust_excess_db: float
    detection_probability: float

    peak_amplitude: float
    clipping_fraction: float

    predicted_max_dbfs: float = math.nan
    compression_db: float = math.nan
    linear: bool = False


# ============================================================
# GÉNÉRATEUR N5183A
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
            raise RuntimeError(f"Instrument inattendu : {identity}")

        print(f"[N5183A] {identity}")
        self.output(False)
        self.dev.write(":FREQ:MODE FIX")
        self.dev.write(":POW:MODE FIX")
        self.check_errors("initialisation")

    def set_frequency(self, frequency_hz: float) -> float:
        self.dev.write(f":FREQ {float(frequency_hz):.3f} HZ")
        actual = float(self.dev.query(":FREQ?"))
        self.check_errors("fréquence")
        return actual

    def set_power(self, power_dbm: float) -> float:
        power_dbm = float(power_dbm)
        if not GENERATOR_MIN_DBM <= power_dbm <= GENERATOR_MAX_DBM:
            raise ValueError(
                f"Puissance générateur interdite : {power_dbm:.2f} dBm"
            )

        self.dev.write(f":POW {power_dbm:.3f} DBM")
        self.check_errors("puissance")
        actual = float(self.dev.query(":POW?"))

        if abs(actual - power_dbm) > 0.1:
            self.output(False)
            raise RuntimeError(
                f"Puissance configurée {actual:.2f} dBm au lieu de "
                f"{power_dbm:.2f} dBm"
            )
        return actual

    def output(self, enabled: bool) -> None:
        self.dev.write(":OUTP ON" if enabled else ":OUTP OFF")
        state = int(float(self.dev.query(":OUTP?")))
        if state != int(enabled):
            raise RuntimeError("Échec RF Output")
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
                f"Erreur SCPI ({operation}) : {' | '.join(errors)}"
            )

    def close(self) -> None:
        try:
            self.output(False)
        finally:
            self.dev.close()
            self.rm.close()


# ============================================================
# USRP
# ============================================================

def configure_usrp(
    usrp,
    center_frequency_hz: float,
    requested_gain_db: float,
) -> float:
    usrp.set_rx_rate(RATE_HZ, CHANNEL)
    usrp.set_rx_antenna(ANTENNA, CHANNEL)
    usrp.set_rx_gain(requested_gain_db, CHANNEL)

    usrp.set_rx_freq(
        uhd.types.TuneRequest(center_frequency_hz, LO_OFFSET_HZ),
        CHANNEL,
    )
    usrp.set_rx_dc_offset(True, CHANNEL)
    usrp.set_rx_iq_balance(True, CHANNEL)

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
        (ACQUISITION_DURATION_S + DISCARD_DURATION_S) * RATE_HZ
    )

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [CHANNEL]
    streamer = usrp.get_rx_stream(stream_args)
    metadata = uhd.types.RXMetadata()

    samples = np.zeros(number_of_samples, dtype=np.complex64)
    buffer = np.zeros(
        (1, streamer.get_max_num_samps()),
        dtype=np.complex64,
    )

    command = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    command.num_samps = number_of_samples
    command.stream_now = True
    streamer.issue_stream_cmd(command)

    total = 0
    while total < number_of_samples:
        received = streamer.recv(buffer, metadata, 3.0)

        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            raise RuntimeError(metadata.strerror())
        if received <= 0:
            raise RuntimeError("Aucun échantillon reçu")

        count = min(received, number_of_samples - total)
        samples[total:total + count] = buffer[0, :count]
        total += count

    discard = int(DISCARD_DURATION_S * RATE_HZ)
    return samples[discard:]


# ============================================================
# CALCUL MAX : IDENTIQUE À TON CODE DE SPECTRE
# ============================================================

def amplitude_to_dbfs(amplitude: float) -> float:
    return float(
        20.0 * math.log10(
            max(float(amplitude), math.sqrt(EPSILON))
        )
    )


def temporal_max_metrics(samples: np.ndarray) -> dict:
    samples = np.asarray(samples, dtype=np.complex64).ravel()
    if samples.size == 0:
        raise ValueError("Acquisition vide")

    if REMOVE_DC:
        samples = samples - np.mean(samples)

    cutoff = min(LP_CUTOFF_HZ, 0.45 * RATE_HZ)
    sos = butter(
        4,
        cutoff / (RATE_HZ / 2.0),
        btype="low",
        output="sos",
    )

    samples_filtered = sosfiltfilt(sos, samples)
    envelope = np.abs(samples_filtered)

    max_amplitude = float(np.max(envelope))
    robust_amplitude = float(
        np.percentile(envelope, ROBUST_PERCENTILE)
    )
    median_amplitude = float(np.median(envelope))

    raw_magnitude = np.abs(samples)
    clipping_fraction = float(
        np.mean(raw_magnitude >= CLIPPING_AMPLITUDE)
    )

    return {
        "max_amplitude": max_amplitude,
        "max_dbfs": amplitude_to_dbfs(max_amplitude),
        "robust_amplitude": robust_amplitude,
        "robust_dbfs": amplitude_to_dbfs(robust_amplitude),
        "median_amplitude": median_amplitude,
        "median_dbfs": amplitude_to_dbfs(median_amplitude),
        "clipping_fraction": clipping_fraction,
    }


def measure_noise(usrp) -> tuple[float, float]:
    max_values = []
    robust_values = []

    for _ in range(NOISE_MEASUREMENTS):
        metrics = temporal_max_metrics(acquire_samples(usrp))
        max_values.append(metrics["max_dbfs"])
        robust_values.append(metrics["robust_dbfs"])

    return (
        float(np.median(max_values)),
        float(np.median(robust_values)),
    )


# ============================================================
# SÉCURITÉ
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
            "\nARRÊT DE SÉCURITÉ\n"
            f"Générateur : {generator_dbm:.2f} dBm\n"
            f"Atténuation : {attenuation_db:.2f} dB\n"
            f"Entrée RX2 : {input_rx2_dbm:.2f} dBm\n"
            f"Limite sûre : {RX2_SAFE_MAX_DBM:.2f} dBm\n"
            f"Seuil dangereux : {RX2_DANGEROUS_DBM:.2f} dBm"
        )

    return input_rx2_dbm


def generator_power_grid(
    attenuation_db: float,
) -> np.ndarray:
    safe_max = min(
        GENERATOR_MAX_DBM,
        RX2_SAFE_MAX_DBM + attenuation_db,
    )

    if safe_max < GENERATOR_MIN_DBM:
        return np.asarray([], dtype=float)

    return np.arange(
        GENERATOR_MIN_DBM,
        safe_max + GENERATOR_STEP_DB / 2.0,
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
            f"Aucune puissance sûre avec {attenuation_db:.1f} dB"
        )

    print("\n" + "#" * 76)
    print(f"INSTALLE {attenuation_db:.1f} dB D'ATTÉNUATION")
    print("Montage : N5183A -> atténuateur(s) -> RX2")
    print(f"Gains mesurés : {GAINS_DB}")
    print(
        f"Plage RX2 : {grid[0] - attenuation_db:.1f} à "
        f"{grid[-1] - attenuation_db:.1f} dBm"
    )
    print("La sortie RF est OFF")

    answer = input(
        "Tape exactement OUI après vérification : "
    ).strip().upper()

    if answer != "OUI":
        raise RuntimeError("Mesure annulée")


# ============================================================
# MESURE D'UN POINT
# ============================================================

def measure_generator_point(
    usrp,
    generator: N5183A,
    center_frequency_hz: float,
    generator_frequency_hz: float,
    gain_db: float,
    attenuation_db: float,
    requested_generator_dbm: float,
    noise_max_dbfs: float,
    noise_robust_dbfs: float,
) -> list[RawMeasurement]:
    assert_safe_power(requested_generator_dbm, attenuation_db)

    actual_generator_dbm = generator.set_power(
        requested_generator_dbm
    )
    input_rx2_dbm = assert_safe_power(
        actual_generator_dbm,
        attenuation_db,
    )

    time.sleep(SETTLING_TIME_S)
    rows = []

    for _ in range(MEASUREMENTS_PER_POINT):
        metrics = temporal_max_metrics(acquire_samples(usrp))

        max_excess_db = metrics["max_dbfs"] - noise_max_dbfs
        robust_excess_db = (
            metrics["robust_dbfs"] - noise_robust_dbfs
        )

        detected = bool(
            max_excess_db >= MIN_MAX_EXCESS_DB
            and robust_excess_db >= MIN_ROBUST_EXCESS_DB
        )

        rows.append(
            RawMeasurement(
                frequency_hz=float(center_frequency_hz),
                frequency_mhz=float(center_frequency_hz / 1e6),
                generator_frequency_hz=float(generator_frequency_hz),
                tone_offset_hz=float(TEST_TONE_OFFSET_HZ),
                gain_db=float(gain_db),
                attenuation_db=float(attenuation_db),
                generator_dbm=float(actual_generator_dbm),
                input_rx2_dbm=float(input_rx2_dbm),

                max_amplitude=float(metrics["max_amplitude"]),
                max_dbfs=float(metrics["max_dbfs"]),
                robust_amplitude=float(metrics["robust_amplitude"]),
                robust_dbfs=float(metrics["robust_dbfs"]),
                median_amplitude=float(metrics["median_amplitude"]),
                median_dbfs=float(metrics["median_dbfs"]),

                noise_max_dbfs=float(noise_max_dbfs),
                noise_robust_dbfs=float(noise_robust_dbfs),
                max_excess_db=float(max_excess_db),
                robust_excess_db=float(robust_excess_db),

                clipping_fraction=float(
                    metrics["clipping_fraction"]
                ),
                detected=detected,
            )
        )

    return rows


# ============================================================
# AGRÉGATION ET DYNAMIQUE
# ============================================================

def aggregate_measurements(
    measurements: list[RawMeasurement],
) -> list[AggregatedPoint]:
    grouped = {}

    for row in measurements:
        key = (
            round(row.frequency_hz, 1),
            round(row.gain_db, 3),
            round(row.input_rx2_dbm, 3),
        )
        grouped.setdefault(key, []).append(row)

    points = []

    for (frequency_hz, gain_db, input_rx2_dbm), group in grouped.items():
        points.append(
            AggregatedPoint(
                frequency_hz=float(frequency_hz),
                frequency_mhz=float(frequency_hz / 1e6),
                gain_db=float(gain_db),
                input_rx2_dbm=float(input_rx2_dbm),
                generator_dbm=float(
                    np.median([m.generator_dbm for m in group])
                ),
                attenuation_db=float(
                    np.median([m.attenuation_db for m in group])
                ),
                max_dbfs=float(
                    np.median([m.max_dbfs for m in group])
                ),
                robust_dbfs=float(
                    np.median([m.robust_dbfs for m in group])
                ),
                median_dbfs=float(
                    np.median([m.median_dbfs for m in group])
                ),
                noise_max_dbfs=float(
                    np.median([m.noise_max_dbfs for m in group])
                ),
                noise_robust_dbfs=float(
                    np.median([m.noise_robust_dbfs for m in group])
                ),
                median_max_excess_db=float(
                    np.median([m.max_excess_db for m in group])
                ),
                median_robust_excess_db=float(
                    np.median([m.robust_excess_db for m in group])
                ),
                detection_probability=float(
                    np.mean([m.detected for m in group])
                ),
                peak_amplitude=float(
                    np.max([m.max_amplitude for m in group])
                ),
                clipping_fraction=float(
                    np.max([m.clipping_fraction for m in group])
                ),
            )
        )

    return sorted(
        points,
        key=lambda p: (
            p.gain_db,
            p.frequency_hz,
            p.input_rx2_dbm,
        ),
    )


def fit_linear_region(
    points: list[AggregatedPoint],
) -> Optional[tuple[float, float]]:
    fit_points = [
        p
        for p in points
        if (
            p.median_max_excess_db >= LINEAR_FIT_MIN_EXCESS_DB
            and p.max_dbfs <= LINEAR_FIT_MAX_DBFS
            and p.clipping_fraction <= MAX_CLIPPING_FRACTION
        )
    ]

    if len(fit_points) < 3:
        return None

    x = np.asarray(
        [p.input_rx2_dbm for p in fit_points],
        dtype=float,
    )
    y = np.asarray(
        [p.max_dbfs for p in fit_points],
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
        point.predicted_max_dbfs = (
            slope * point.input_rx2_dbm + intercept
        )
        point.compression_db = (
            point.predicted_max_dbfs - point.max_dbfs
        )

        point.linear = bool(
            point.detection_probability
            >= MIN_DETECTION_PROBABILITY
            and point.max_dbfs <= MAX_ACCEPTED_MAX_DBFS
            and point.clipping_fraction
            <= MAX_CLIPPING_FRACTION
            and abs(point.compression_db)
            <= MAX_COMPRESSION_DB
        )


def build_summary(
    points: list[AggregatedPoint],
) -> list[dict]:
    grouped = {}

    for point in points:
        key = (
            round(point.frequency_hz, 1),
            round(point.gain_db, 3),
        )
        grouped.setdefault(key, []).append(point)

    summary = []

    for (frequency_hz, gain_db), case_points in sorted(grouped.items()):
        case_points = sorted(
            case_points,
            key=lambda p: p.input_rx2_dbm,
        )

        fit = fit_linear_region(case_points)
        evaluate_linearity(case_points, fit)

        detected_points = [
            p for p in case_points
            if p.detection_probability >= MIN_DETECTION_PROBABILITY
        ]
        linear_points = [
            p for p in case_points
            if p.linear
        ]

        pmin = (
            min(p.input_rx2_dbm for p in detected_points)
            if detected_points else None
        )
        pmax = (
            max(p.input_rx2_dbm for p in linear_points)
            if linear_points else None
        )

        pmin_status = (
            "aucun_signal_detecte"
            if pmin is None
            else (
                "limite_basse_du_banc"
                if np.isclose(
                    pmin,
                    min(p.input_rx2_dbm for p in case_points),
                    atol=0.1,
                )
                else "mesure_valide"
            )
        )

        pmax_status = (
            "aucun_point_lineaire"
            if pmax is None
            else (
                "limite_haute_du_banc"
                if np.isclose(
                    pmax,
                    max(p.input_rx2_dbm for p in case_points),
                    atol=0.1,
                )
                else "mesure_valide"
            )
        )

        dynamic = (
            pmax - pmin
            if pmin is not None and pmax is not None
            else None
        )

        summary.append(
            {
                "frequency_hz": float(frequency_hz),
                "frequency_mhz": float(frequency_hz / 1e6),
                "gain_db": float(gain_db),
                "rx2_min_tested_dbm": float(
                    min(p.input_rx2_dbm for p in case_points)
                ),
                "rx2_max_tested_dbm": float(
                    max(p.input_rx2_dbm for p in case_points)
                ),
                "noise_max_dbfs": float(
                    np.median([p.noise_max_dbfs for p in case_points])
                ),
                "noise_robust_dbfs": float(
                    np.median([
                        p.noise_robust_dbfs for p in case_points
                    ])
                ),
                "minimum_detectable_dbm": (
                    "" if pmin is None else float(pmin)
                ),
                "pmin_status": pmin_status,
                "maximum_linear_dbm": (
                    "" if pmax is None else float(pmax)
                ),
                "pmax_status": pmax_status,
                "dynamic_range_db": (
                    "" if dynamic is None else float(dynamic)
                ),
                "linear_fit_slope": (
                    "" if fit is None else float(fit[0])
                ),
                "linear_fit_intercept": (
                    "" if fit is None else float(fit[1])
                ),
                "number_of_input_points": len(case_points),
            }
        )

    return summary


# ============================================================
# SAUVEGARDE
# ============================================================

def save_dataclass_rows(path: Path, rows: list) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(asdict(rows[0]).keys()),
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def save_dict_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# GRAPHES
# ============================================================

def save_all_plots(points_csv: Path, summary_csv: Path) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    points_df = pd.read_csv(points_csv)
    summary_df = pd.read_csv(summary_csv)

    points_df = points_df.sort_values(
        ["gain_db", "frequency_mhz", "input_rx2_dbm"]
    )
    summary_df = summary_df.sort_values(
        ["gain_db", "frequency_mhz"]
    )

    gains = sorted(points_df["gain_db"].dropna().unique())

    # Noise floor MAX + niveaux injectés
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
            gain_summary["noise_max_dbfs"],
            linestyle="--",
            linewidth=2.5,
            label="Noise floor — MAX temporel",
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
                level_data["max_dbfs"],
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=f"Entrée RX2 = {input_level_dbm:.0f} dBm",
            )

        plt.title(
            f"B200 — MAX temporel selon la fréquence "
            f"(gain RX = {gain_db:.0f} dB)"
        )
        plt.xlabel("Fréquence centrale (MHz)")
        plt.ylabel("Amplitude maximale mesurée (dBFS)")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR
            / f"gain_{gain_db:.0f}_noise_and_max_measured.png",
            dpi=180,
        )
        plt.close()

    # Pmin / Pmax
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
            f"Dynamique fondée sur le MAX temporel "
            f"(gain RX = {gain_db:.0f} dB)"
        )
        plt.xlabel("Fréquence centrale (MHz)")
        plt.ylabel("Puissance à l'entrée RX2 (dBm)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR
            / f"gain_{gain_db:.0f}_max_dynamic_limits.png",
            dpi=180,
        )
        plt.close()

    # Dynamique tous gains
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

    plt.title("Dynamique du B200 fondée sur le MAX temporel")
    plt.xlabel("Fréquence centrale (MHz)")
    plt.ylabel("Dynamique d'entrée (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "dynamic_range_max_all_gains.png",
        dpi=180,
    )
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Nouveau dossier de résultats :")
    print(OUTPUT_DIR)

    usrp = uhd.usrp.MultiUSRP(
        f"serial={USRP_SERIAL}"
    )
    generator = N5183A(GENERATOR_IP)

    all_measurements: list[RawMeasurement] = []

    try:
        for attenuation_db in ATTENUATIONS_DB:
            attenuation_db = float(attenuation_db)

            request_attenuator_confirmation(
                generator,
                attenuation_db,
            )

            grid = generator_power_grid(attenuation_db)

            for center_frequency_hz in FREQUENCIES_HZ:
                center_frequency_hz = float(center_frequency_hz)

                generator_frequency_hz = (
                    center_frequency_hz + TEST_TONE_OFFSET_HZ
                )

                generator.output(False)
                actual_generator_frequency_hz = (
                    generator.set_frequency(generator_frequency_hz)
                )

                for requested_gain_db in GAINS_DB:
                    print("\n" + "=" * 76)
                    print(
                        f"fc={center_frequency_hz / 1e6:.1f} MHz | "
                        f"fgen={actual_generator_frequency_hz / 1e6:.1f} MHz | "
                        f"gain={requested_gain_db:.1f} dB | "
                        f"att={attenuation_db:.1f} dB"
                    )

                    actual_gain_db = configure_usrp(
                        usrp,
                        center_frequency_hz,
                        requested_gain_db,
                    )

                    generator.output(False)
                    time.sleep(SETTLING_TIME_S)

                    (
                        noise_max_dbfs,
                        noise_robust_dbfs,
                    ) = measure_noise(usrp)

                    print(
                        f"Bruit MAX = {noise_max_dbfs:.2f} dBFS | "
                        f"Bruit robuste = {noise_robust_dbfs:.2f} dBFS"
                    )

                    generator.set_power(float(grid[0]))
                    assert_safe_power(
                        float(grid[0]),
                        attenuation_db,
                    )
                    generator.output(True)

                    try:
                        for generator_dbm in grid:
                            rows = measure_generator_point(
                                usrp=usrp,
                                generator=generator,
                                center_frequency_hz=center_frequency_hz,
                                generator_frequency_hz=actual_generator_frequency_hz,
                                gain_db=actual_gain_db,
                                attenuation_db=attenuation_db,
                                requested_generator_dbm=float(generator_dbm),
                                noise_max_dbfs=noise_max_dbfs,
                                noise_robust_dbfs=noise_robust_dbfs,
                            )

                            all_measurements.extend(rows)

                            print(
                                f"Gen={generator_dbm:6.1f} dBm | "
                                f"RX2={generator_dbm - attenuation_db:7.1f} dBm | "
                                f"Max={np.median([r.max_dbfs for r in rows]):7.2f} dBFS | "
                                f"Robuste={np.median([r.robust_dbfs for r in rows]):7.2f} dBFS | "
                                f"Pd={np.mean([r.detected for r in rows]):.2f}"
                            )

                            # Sauvegarde progressive
                            save_dataclass_rows(
                                RAW_CSV,
                                all_measurements,
                            )

                    finally:
                        generator.output(False)

    finally:
        generator.close()

    print("\nAgrégation...")
    aggregated_points = aggregate_measurements(
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
    print(f"CSV brut   : {RAW_CSV}")
    print(f"CSV agrégé : {POINTS_CSV}")
    print(f"CSV résumé : {SUMMARY_CSV}")
    print(f"Graphes    : {PLOTS_DIR}")


if __name__ == "__main__":
    main()