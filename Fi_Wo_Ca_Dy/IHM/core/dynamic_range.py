#!/usr/bin/env python3
"""
core/dynamic_range.py
---------------------
Table de dynamique par fréquence et par gain.

Construit Pmin(f, G) et Pmax(f, G) en dBm à partir du fichier de calibration :

    Pmax(f, G) = SAT_DBFS + C(f, G)      # seuil de saturation ADC → dBm
    Pmin(f, G) = NOISE_FLOOR_DBFS + C(f, G)   # plancher de bruit → dBm

Logique AGC (cf. flowchart) :
    - Si Pmes proche de Pmin  → augmenter le gain
    - Si Pmes proche de Pmax  → diminuer le gain
"""
from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────
# Constantes physiques
# ──────────────────────────────────────────────────────────────

# Seuil de saturation de l'ADC en dBFS (avant saturation réelle à 0 dBFS)
# Marge de sécurité de 3 dB
SAT_DBFS: float = -3.0

# Plancher de bruit interne estimé du B200 (ADC 12-bits, BW 12 MHz)
# Mesuré à ~-65 dBFS sur les graphes de dynamique du PDF
NOISE_FLOOR_DBFS: float = -65.0

# Marge de garde en dBm autour de Pmin et Pmax pour le changement de gain
# Si Pmes < Pmin + MARGIN_DB → signal faible → monter le gain
# Si Pmes > Pmax - MARGIN_DB → signal fort  → baisser le gain
MARGIN_DB: float = 8.0


# ──────────────────────────────────────────────────────────────
# Construction de la table Pmin / Pmax
# ──────────────────────────────────────────────────────────────

def build_dynamic_range_table(calibration_table: dict) -> dict:
    """
    Construit la table de dynamique à partir de la table de calibration.

    Paramètre :
        calibration_table : dict  { freq_hz : { gain_db : C(f,G) } }

    Retourne :
        {
            gain_db: {
                "frequencies_hz": np.ndarray,
                "pmin_dbm":       np.ndarray,   # plancher bruit en dBm
                "pmax_dbm":       np.ndarray,   # seuil saturation en dBm
            },
            ...
        }
    """
    gains = sorted({
        float(g)
        for gt in calibration_table.values()
        for g in gt.keys()
    })

    result: dict = {}

    for gain_db in gains:
        freqs, pmins, pmaxs = [], [], []
        for freq_hz in sorted(calibration_table.keys()):
            gt = calibration_table[freq_hz]
            if float(gain_db) not in gt:
                continue
            C = float(gt[float(gain_db)])
            freqs.append(float(freq_hz))
            pmins.append(NOISE_FLOOR_DBFS + C)
            pmaxs.append(SAT_DBFS + C)

        if not freqs:
            continue

        result[float(gain_db)] = {
            "frequencies_hz": np.asarray(freqs, dtype=float),
            "pmin_dbm":       np.asarray(pmins, dtype=float),
            "pmax_dbm":       np.asarray(pmaxs, dtype=float),
        }

    return result


def get_pmin_pmax(
    dynamic_table: dict,
    gain_db: float,
    frequency_hz: float,
) -> tuple[float, float]:
    """
    Retourne (Pmin_dBm, Pmax_dBm) pour un gain et une fréquence donnés.

    Utilise une interpolation linéaire entre les points de calibration.

    Raises ValueError si le gain est absent de la table.
    """
    # Trouver le gain le plus proche dans la table
    available = np.asarray(sorted(dynamic_table.keys()), dtype=float)
    nearest_gain = float(available[np.argmin(np.abs(available - gain_db))])

    if nearest_gain not in dynamic_table:
        raise ValueError(f"Gain {gain_db} dB absent de la table de dynamique.")

    entry = dynamic_table[nearest_gain]
    freqs  = entry["frequencies_hz"]
    pmin_a = entry["pmin_dbm"]
    pmax_a = entry["pmax_dbm"]

    # Interpolation linéaire (extrapolation par clip aux bords)
    pmin = float(np.interp(frequency_hz, freqs, pmin_a))
    pmax = float(np.interp(frequency_hz, freqs, pmax_a))

    return pmin, pmax


# ──────────────────────────────────────────────────────────────
# Décision de gain basée sur Pmin / Pmax
# ──────────────────────────────────────────────────────────────

def decide_gain_from_dynamic(
    dynamic_table: dict,
    current_gain_db: float,
    measured_dbm: float,
    frequency_hz: float,
    gains_list: list[float],
) -> tuple[float | None, str]:
    """
    Décide si le gain doit changer en comparant Pmes à Pmin et Pmax.

    Algorithme (cf. flowchart) :
        - Pmes proche de Pmin → signal faible → augmenter le gain
        - Pmes proche de Pmax → signal fort   → diminuer le gain
        - Sinon               → gain optimal

    Retourne :
        (nouveau_gain_db, raison_str)
        None si aucun changement nécessaire.
    """
    try:
        pmin, pmax = get_pmin_pmax(dynamic_table, current_gain_db, frequency_hz)
    except ValueError:
        return None, "OK"

    # Cas 1 : saturation réelle (clipping dBFS > seuil) — déjà géré upstream
    # Ici on travaille en dBm

    # Cas 2 : Pmes proche de Pmax → risque saturation → baisser le gain
    if measured_dbm > pmax - MARGIN_DB:
        gains_arr = np.asarray(gains_list, dtype=float)
        idx = int(np.argmin(np.abs(gains_arr - current_gain_db)))
        if idx > 0:
            next_g = float(gains_arr[idx - 1])
            return next_g, f"Pmes={measured_dbm:.1f}dBm > Pmax-margin={pmax-MARGIN_DB:.1f}dBm → gain↓{next_g:.0f}dB"
        else:
            return None, "SAT@MIN_GAIN"

    # Cas 3 : Pmes proche de Pmin → signal trop faible → augmenter le gain
    if measured_dbm < pmin + MARGIN_DB:
        gains_arr = np.asarray(gains_list, dtype=float)
        idx = int(np.argmin(np.abs(gains_arr - current_gain_db)))
        if idx < len(gains_arr) - 1:
            next_g = float(gains_arr[idx + 1])
            return next_g, f"Pmes={measured_dbm:.1f}dBm < Pmin+margin={pmin+MARGIN_DB:.1f}dBm → gain↑{next_g:.0f}dB"
        else:
            return None, "WEAK@MAX_GAIN"

    # Cas 4 : gain optimal
    return None, "OK"
