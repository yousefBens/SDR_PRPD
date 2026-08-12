import pickle
from pathlib import Path


CAL_PATH = (
    Path.home()
    / "b200_calibration_results"
    / "rx2_complete_100MHz_2GHz.pickle"
)


with CAL_PATH.open("rb") as file:
    results = pickle.load(file)


table = results[0]["RX2"]


print(
    f"{'Fréquence MHz':>15} | "
    f"{'Gain réel dB':>13} | "
    f"{'Référence 0 dBFS':>20}"
)

print("-" * 56)


for frequency_hz in sorted(table):
    gain_table = table[frequency_hz]

    for gain_db in sorted(
        gain_table,
        reverse=True,
    ):
        reference_dbm = gain_table[gain_db]

        print(
            f"{frequency_hz / 1e6:15.1f} | "
            f"{gain_db:13.1f} | "
            f"{reference_dbm:17.2f} dBm"
        )
