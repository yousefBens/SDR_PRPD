#!/usr/bin/env python3
from __future__ import annotations

import csv
import pickle
import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION -- MODIFIER SEULEMENT CETTE PARTIE
# ============================================================

USRP_SERIAL = "306BD15"

START_HZ = 80e6
STOP_HZ = 5.9e9
STEP_HZ = 50e6

SAMPLE_RATE = 12e6

CHANNEL = 0
ANTENNA = "RX2"

GENERATOR_IP = "192.168.1.100"
GENERATOR_MIN_DBM = -20.0
GENERATOR_MAX_DBM = +20.0

RX2_DANGEROUS_DBM = -15.0
RX2_SAFE_MAX_DBM = -20.0

GROUPS = [
    {"attenuation_db": 60.0, "gains_db": [60.0, 76.0]},
    {"attenuation_db": 40.0, "gains_db": [20.0, 40.0]},
    {"attenuation_db": 20.0, "gains_db": [0.0]},
]

TARGET_DBFS_CANDIDATES = [
    -6.0,
    -12.0,
    -18.0,
    -24.0,
    -30.0,
    -36.0,
    -42.0,
    -48.0,
]

TARGET_LOWER_MARGIN_DB = 8.0
TARGET_UPPER_MARGIN_DB = 6.0

FINAL_PICKLE = "rx2_power_calibration_final.pickle"
FINAL_CSV = "rx2_power_calibration_final.csv"
METADATA_CSV = "rx2_power_calibration_runs.csv"


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CAL_DIR = SCRIPT_DIR / "Cal_Files"
ENGINE = SCRIPT_DIR / "uhd_power_cal_adapted.py"

CAL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HARDWARE LIMITS
# ============================================================

def max_safe_generator_power(attenuation_db):
    return float(
        min(
            GENERATOR_MAX_DBM,
            RX2_SAFE_MAX_DBM + attenuation_db,
        )
    )


def min_rx2_power(attenuation_db):
    return float(
        GENERATOR_MIN_DBM - attenuation_db
    )


def max_rx2_power(attenuation_db):
    return float(
        max_safe_generator_power(attenuation_db)
        - attenuation_db
    )


def validate_config():
    if not ENGINE.exists():
        raise FileNotFoundError(f"Missing engine: {ENGINE}")

    if RX2_SAFE_MAX_DBM >= RX2_DANGEROUS_DBM:
        raise ValueError(
            "RX2_SAFE_MAX_DBM must be below RX2_DANGEROUS_DBM"
        )

    if START_HZ <= 0 or STOP_HZ <= START_HZ:
        raise ValueError("Invalid frequency range")

    if STEP_HZ <= 0:
        raise ValueError("STEP_HZ must be > 0")


# ============================================================
# RUN ONE GAIN
# ============================================================

def run_gain(
    attenuation_db,
    gain_db,
    target_dbfs,
    output_path,
):
    generator_max = max_safe_generator_power(
        attenuation_db
    )

    rx2_min = min_rx2_power(
        attenuation_db
    )

    start_input_power = max(
        -70.0,
        rx2_min,
    )

    lower = target_dbfs - TARGET_LOWER_MARGIN_DB
    upper = min(
        target_dbfs + TARGET_UPPER_MARGIN_DB,
        -1.0,
    )

    if output_path.exists():
        output_path.unlink()

    command = [
        sys.executable,
        str(ENGINE),

        f"--args=serial={USRP_SERIAL}",

        "--dir", "rx",

        "--start", str(START_HZ),
        "--stop", str(STOP_HZ),
        "--step", str(STEP_HZ),

        "--gains", str(gain_db),
        "--gain-step", "1",

        "--min-input-power", str(start_input_power),

        "--target-dbfs", str(target_dbfs),
        "--dbfs-lower-limit", str(lower),
        "--dbfs-upper-limit", str(upper),

        "--antenna", ANTENNA,
        "--channels", str(CHANNEL),
        "--rate", str(SAMPLE_RATE),

        "--attenuation", str(attenuation_db),

        "--meas-dev", "n5183a",

        "-o", "import=n5183a_generator",
        "-o", f"ip={GENERATOR_IP}",
        "-o", f"min_generator_dbm={GENERATOR_MIN_DBM}",
        "-o", f"max_generator_dbm={generator_max}",
        "-o", f"max_dut_input_dbm={RX2_SAFE_MAX_DBM}",

        "--switch", "manual",
        
        "--switch-option", "mode=auto",
        
        "--store", str(output_path),
    ]

    print()
    print(
        f"--- Gain {gain_db:.1f} dB | "
        f"target {target_dbfs:.1f} dBFS ---"
    )
    print(
        f"N5183A autorisé : "
        f"{GENERATOR_MIN_DBM:.1f} -> "
        f"{generator_max:.1f} dBm"
    )
    print(
        f"RX2 accessible : "
        f"{min_rx2_power(attenuation_db):.1f} -> "
        f"{max_rx2_power(attenuation_db):.1f} dBm"
    )

    process = subprocess.Popen(
        command,
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    logs = []

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")
        logs.append(line)

    return_code = process.wait()

    return return_code, "".join(logs)


# ============================================================
# TARGET SEARCH
# ============================================================

def calibrate_gain(
    attenuation_db,
    gain_db,
):
    output_path = (
        CAL_DIR
        / f"rx2_att{attenuation_db:.0f}_gain{gain_db:.0f}.pickle"
    )

    last_log = ""

    for target in TARGET_DBFS_CANDIDATES:
        code, log = run_gain(
            attenuation_db,
            gain_db,
            target,
            output_path,
        )

        last_log = log

        if code == 0 and output_path.exists():
            print(
                f"\n[OK] Gain {gain_db:.1f} dB "
                f"calibrated with target {target:.1f} dBFS"
            )
            return output_path, target

        print(
            f"\n[RETRY] Target {target:.1f} dBFS "
            f"not reachable for gain {gain_db:.1f} dB"
        )

    raise RuntimeError(
        f"Unable to calibrate gain {gain_db:.1f} dB "
        f"with available hardware.\n"
        f"Last log:\n{last_log[-2500:]}"
    )


# ============================================================
# MERGE
# ============================================================

def merge_pickles(paths):
    merged = {}

    for path in paths:
        with path.open("rb") as file:
            results = pickle.load(file)

        table = results[CHANNEL][ANTENNA]

        for frequency_hz, gain_table in table.items():
            frequency_hz = float(frequency_hz)

            merged.setdefault(
                frequency_hz,
                {},
            )

            for gain_db, reference_dbm in gain_table.items():
                gain_db = float(gain_db)

                if gain_db > 76.0 + 1e-6:
                    raise RuntimeError(
                        f"Invalid gain in final table: {gain_db:.2f} dB"
                    )

                merged[
                    frequency_hz
                ][
                    gain_db
                ] = float(reference_dbm)

    final_results = {
        CHANNEL: {
            ANTENNA: merged
        }
    }

    output = CAL_DIR / FINAL_PICKLE

    with output.open("wb") as file:
        pickle.dump(
            final_results,
            file,
        )

    return output, merged


def export_final_csv(merged):
    output = CAL_DIR / FINAL_CSV

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "frequency_hz",
                "frequency_mhz",
                "gain_db",
                "reference_dbm",
            ]
        )

        for frequency_hz in sorted(merged):
            for gain_db in sorted(
                merged[frequency_hz]
            ):
                writer.writerow(
                    [
                        frequency_hz,
                        frequency_hz / 1e6,
                        gain_db,
                        merged[
                            frequency_hz
                        ][
                            gain_db
                        ],
                    ]
                )

    return output


def export_metadata(rows):
    output = CAL_DIR / METADATA_CSV

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "attenuation_db",
                "gain_db",
                "target_dbfs_used",
                "rx2_min_dbm",
                "rx2_max_dbm",
                "generator_min_dbm",
                "generator_max_dbm",
                "pickle_file",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return output


# ============================================================
# MAIN
# ============================================================

def main():
    validate_config()

    print("=" * 78)
    print("B200 RX2 - UHD POWER CALIBRATION ADAPTED TO HARDWARE")
    print("=" * 78)

    print(f"USRP serial    : {USRP_SERIAL}")
    print(
        f"RF range       : "
        f"{START_HZ/1e6:.1f} -> "
        f"{STOP_HZ/1e6:.1f} MHz"
    )
    print(
        f"RF step        : "
        f"{STEP_HZ/1e6:.1f} MHz"
    )
    print(
        f"Sample rate    : "
        f"{SAMPLE_RATE/1e6:.3f} Msps"
    )
    print(
        f"N5183A physical: "
        f"{GENERATOR_MIN_DBM:.1f} -> "
        f"{GENERATOR_MAX_DBM:.1f} dBm"
    )
    print(
        f"RX2 safe max   : "
        f"{RX2_SAFE_MAX_DBM:.1f} dBm"
    )
    print(
        f"RX2 danger     : "
        f"{RX2_DANGEROUS_DBM:.1f} dBm"
    )

    partial_pickles = []
    metadata = []

    for group in GROUPS:
        attenuation = float(
            group["attenuation_db"]
        )

        generator_max = max_safe_generator_power(
            attenuation
        )

        print()
        print("#" * 78)
        print(
            f"INSTALLE EXACTEMENT "
            f"{attenuation:.0f} dB D'ATTÉNUATION"
        )
        print("#" * 78)
        print(f"Gains : {group['gains_db']}")
        print(
            f"N5183A autorisé : "
            f"{GENERATOR_MIN_DBM:.1f} -> "
            f"{generator_max:.1f} dBm"
        )
        print(
            f"RX2 accessible : "
            f"{min_rx2_power(attenuation):.1f} -> "
            f"{max_rx2_power(attenuation):.1f} dBm"
        )

        answer = input(
            "Tape exactement OUI après vérification : "
        ).strip().upper()

        if answer != "OUI":
            raise RuntimeError("Calibration cancelled")

        for gain in group["gains_db"]:
            path, target = calibrate_gain(
                attenuation,
                float(gain),
            )

            partial_pickles.append(path)

            metadata.append(
                {
                    "attenuation_db": attenuation,
                    "gain_db": float(gain),
                    "target_dbfs_used": target,
                    "rx2_min_dbm": min_rx2_power(
                        attenuation
                    ),
                    "rx2_max_dbm": max_rx2_power(
                        attenuation
                    ),
                    "generator_min_dbm": GENERATOR_MIN_DBM,
                    "generator_max_dbm": generator_max,
                    "pickle_file": path.name,
                }
            )

    print()
    print("=" * 78)
    print("MERGING UHD CALIBRATION TABLES")
    print("=" * 78)

    final_pickle, merged = merge_pickles(
        partial_pickles
    )

    final_csv = export_final_csv(
        merged
    )

    metadata_csv = export_metadata(
        metadata
    )

    gains = sorted(
        {
            gain
            for gain_table in merged.values()
            for gain in gain_table
        }
    )

    print()
    print(
        "Final gains : "
        + ", ".join(
            f"{gain:.1f}"
            for gain in gains
        )
    )
    print(f"Final pickle : {final_pickle}")
    print(f"Final CSV    : {final_csv}")
    print(f"Run metadata : {metadata_csv}")
    print("Calibration completed.")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nCalibration interrupted.")
        sys.exit(130)

    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
