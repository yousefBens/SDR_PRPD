import pickle
import numpy as np
from pathlib import Path

# ============================================================
# Charger la table de calibration
# ============================================================

CAL_FILE = Path.home() / "b200_calibration_results" / "rx2_complete_100MHz_2GHz.pickle"

with open(CAL_FILE, "rb") as f:
    cal = pickle.load(f)

cal = cal[0]["RX2"]


# ============================================================
# Interpolation de la référence
# ============================================================

def get_reference(freq_hz, gain_db):
    gain_db = float(gain_db)

    freqs = []
    refs = []

    for f in sorted(cal.keys()):
        if gain_db in cal[f]:
            freqs.append(float(f))
            refs.append(float(cal[f][gain_db]))

    freqs = np.array(freqs)
    refs = np.array(refs)

    if freq_hz < freqs.min() or freq_hz > freqs.max():
        raise ValueError("Fréquence hors de la plage calibrée.")

    return np.interp(freq_hz, freqs, refs)


# ============================================================
# Conversion
# ============================================================

def dbfs_to_dbm(dbfs, freq_hz, gain_db):

    ref = get_reference(freq_hz, gain_db)

    dbm = dbfs + ref

    return dbm, ref


# ============================================================
# Interface utilisateur
# ============================================================

freq = float(input("Fréquence (MHz) : ")) * 1e6
gain = float(input("Gain (dB) : "))
dbfs = float(input("Mesure (dBFS) : "))

dbm, ref = dbfs_to_dbm(dbfs, freq, gain)

print("\n-----------------------------")
print(f"Référence 0 dBFS : {ref:.2f} dBm")
print(f"Mesure          : {dbfs:.2f} dBFS")
print(f"Puissance RF    : {dbm:.2f} dBm")
print("-----------------------------")