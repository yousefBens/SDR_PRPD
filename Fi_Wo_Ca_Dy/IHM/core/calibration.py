#!/usr/bin/env python3
"""
core/calibration.py
-------------------
Fonctions de calibration extraites de spec_time_max_calib.py.
Supporte tous les gains : 0, 20, 40, 60, 76 dB.

Conversion : P_RX2_dBm = P_dBFS + C(f, G)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────

CAL_FILE = Path(
    "/home/yousef/Documents/testing_scripts/GEVernova/"
    "Fi_Wo_Ca_Dy/Calib/Cal_Files/"
    "rx2_power_calibration_final.pickle"
)

ANTENNA = "RX2"
GAINS_DB = [0.0, 20.0, 40.0, 60.0, 76.0]

MAX_GAIN_DISTANCE_DB = 1.0
CAL_FREQUENCY_TOLERANCE_HZ = 1e3
EPSILON = 1e-12


# ──────────────────────────────────────────────────────────────
# Chargement
# ──────────────────────────────────────────────────────────────

def load_power_calibration(calibration_file: Path = CAL_FILE) -> dict:
    """
    Charge rx2_power_calibration_final.pickle.

    Structure attendue :
        calibration[0]["RX2"][frequency_hz][gain_db] = reference_dbm
    """
    if not calibration_file.exists():
        raise FileNotFoundError(
            f"Fichier de calibration introuvable :\n{calibration_file}"
        )

    with calibration_file.open("rb") as f:
        calibration = pickle.load(f)

    if 0 not in calibration:
        raise ValueError("Canal 0 absent de la table de calibration.")

    if ANTENNA not in calibration[0]:
        raise ValueError(f"Antenne {ANTENNA} absente de la calibration.")

    raw_table = calibration[0][ANTENNA]

    if not raw_table:
        raise ValueError("La table de calibration est vide.")

    # Normalisation des clés en float
    table: dict = {}
    for frequency_hz, gain_table in raw_table.items():
        table[float(frequency_hz)] = {
            float(g): float(ref) for g, ref in gain_table.items()
        }

    return table


# ──────────────────────────────────────────────────────────────
# Sélection du gain calibré
# ──────────────────────────────────────────────────────────────

def select_calibrated_gain(
    calibration_table: dict,
    requested_gain_db: float,
) -> float:
    """Retourne le gain calibré le plus proche du gain demandé."""
    gains = sorted({
        float(g)
        for gt in calibration_table.values()
        for g in gt.keys()
    })

    if not gains:
        raise ValueError("Aucun gain dans la calibration.")

    arr = np.asarray(gains, dtype=float)
    selected = float(arr[np.argmin(np.abs(arr - requested_gain_db))])
    dist = abs(selected - requested_gain_db)

    if dist > MAX_GAIN_DISTANCE_DB:
        raise ValueError(
            f"Gain demandé {requested_gain_db:.1f} dB, "
            f"plus proche calibré {selected:.1f} dB (écart {dist:.1f} dB)"
        )

    return selected


def build_gain_calibration(
    calibration_table: dict,
    gain_db: float,
) -> dict:
    """
    Extrait fréquences + références dBm pour un gain donné.

    Retourne :
        {
            "gain_db": float,
            "frequencies_hz": np.ndarray,
            "references_dbm": np.ndarray,
        }
    """
    selected_gain = select_calibrated_gain(calibration_table, gain_db)

    freqs, refs = [], []
    for freq_hz in sorted(calibration_table.keys()):
        gt = calibration_table[freq_hz]
        if selected_gain not in gt:
            continue
        freqs.append(float(freq_hz))
        refs.append(float(gt[selected_gain]))

    if not freqs:
        raise ValueError(
            f"Aucun point de calibration pour le gain {selected_gain:.1f} dB."
        )

    return {
        "gain_db": float(selected_gain),
        "frequencies_hz": np.asarray(freqs, dtype=float),
        "references_dbm": np.asarray(refs, dtype=float),
    }


def build_all_gain_calibrations(calibration_table: dict) -> dict[float, dict]:
    """
    Construit les calibrations pour TOUS les gains disponibles.

    Retourne :
        { gain_db: gain_calibration_dict, ... }
    """
    result = {}
    for g in GAINS_DB:
        try:
            result[g] = build_gain_calibration(calibration_table, g)
        except ValueError:
            pass  # Gain non présent dans cette table
    return result


# ──────────────────────────────────────────────────────────────
# Interpolation et conversion
# ──────────────────────────────────────────────────────────────

def interpolate_reference_dbm(
    gain_calibration: dict,
    frequency_hz: float,
) -> float:
    """Interpolation linéaire de la référence dBm à la fréquence demandée."""
    frequency_hz = float(frequency_hz)
    freqs = gain_calibration["frequencies_hz"]
    refs  = gain_calibration["references_dbm"]

    f_min = float(freqs.min())
    f_max = float(freqs.max())

    if frequency_hz < f_min:
        if (f_min - frequency_hz) <= CAL_FREQUENCY_TOLERANCE_HZ:
            frequency_hz = f_min
        else:
            raise ValueError(
                f"Fréquence {frequency_hz/1e6:.3f} MHz < calibration "
                f"({f_min/1e6:.1f} MHz)."
            )

    if frequency_hz > f_max:
        if (frequency_hz - f_max) <= CAL_FREQUENCY_TOLERANCE_HZ:
            frequency_hz = f_max
        else:
            raise ValueError(
                f"Fréquence {frequency_hz/1e6:.3f} MHz > calibration "
                f"({f_max/1e6:.1f} MHz)."
            )

    return float(np.interp(frequency_hz, freqs, refs))


def dbfs_to_dbm(
    gain_calibration: dict,
    measured_dbfs: float,
    frequency_hz: float,
) -> tuple[float, float]:
    """
    Conversion dBFS → dBm.

    Retourne : (estimated_dbm, reference_dbm)
    """
    ref_dbm = interpolate_reference_dbm(gain_calibration, frequency_hz)
    est_dbm = float(measured_dbfs) + ref_dbm
    return float(est_dbm), float(ref_dbm)


# ──────────────────────────────────────────────────────────────
# Utilitaire dBFS
# ──────────────────────────────────────────────────────────────

def amplitude_to_dbfs(amplitude: float) -> float:
    return float(20.0 * np.log10(max(float(amplitude), EPSILON)))
