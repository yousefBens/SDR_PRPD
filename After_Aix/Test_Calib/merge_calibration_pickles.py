#!/usr/bin/env python3

import pickle
from pathlib import Path
from pprint import pprint


BASE_DIR = Path.home() / "b200_calibration_results"

FILES = [
    BASE_DIR / "rx2_att60_gain60_76.pickle",
    BASE_DIR / "rx2_att40_gain20_40.pickle",
    BASE_DIR / "rx2_att20_gain0.pickle",
]

OUTPUT = BASE_DIR / "rx2_complete_100MHz_2GHz.pickle"

CHANNEL = 0
ANTENNA = "RX2"


def normalize_gain(gain_db):
    gain_db = float(gain_db)

    # Le script demande 80 dB,
    # mais le B200 applique réellement 76 dB
    if abs(gain_db - 80.0) < 1e-6:
        return 76.0

    return gain_db


merged_table = {}

for path in FILES:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier absent : {path}"
        )

    with path.open("rb") as file:
        raw_results = pickle.load(file)

    calibration = raw_results[CHANNEL][ANTENNA]

    for frequency_hz, gain_table in calibration.items():
        frequency_hz = float(frequency_hz)

        if frequency_hz not in merged_table:
            merged_table[frequency_hz] = {}

        for gain_db, reference_dbm in gain_table.items():
            actual_gain_db = normalize_gain(gain_db)

            merged_table[frequency_hz][actual_gain_db] = float(
                reference_dbm
            )


merged_results = {
    CHANNEL: {
        ANTENNA: merged_table
    }
}


with OUTPUT.open("wb") as file:
    pickle.dump(
        merged_results,
        file
    )


print("Calibration fusionnée :")
pprint(merged_results)

print()
print("Fichier créé :", OUTPUT)
