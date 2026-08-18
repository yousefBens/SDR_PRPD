#!/usr/bin/env python3
"""
core/auto_gain.py
-----------------

Gestion automatique du gain basée sur la dynamique
réellement caractérisée du B200.

Principe pour chaque fréquence :

    1) Première acquisition avec GAIN_START_DB = 40 dB
    2) Conversion du MAX temporel en dBm
    3) Lecture de :
           Pmin(f, gain)
           Pmax(f, gain)
       depuis dynamic_range_max_summary.csv

    4) Si :
           Pmes < Pmin + marge
       => augmenter le gain

       Si :
           Pmes > Pmax - marge
       => diminuer le gain

       Sinon :
           garder le gain

    5) Si le gain change :
           refaire UNE acquisition à la même fréquence

    6) Ensuite passer à la fréquence suivante.

Important :
    maximum 2 acquisitions par fréquence :
        acquisition initiale
        + éventuellement une acquisition avec gain corrigé
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1) CONFIGURATION
# ============================================================

GAINS_DB: list[float] = [
    0.0,
    20.0,
    40.0,
    60.0,
    76.0,
]

GAIN_START_DB: float = 40.0


# ------------------------------------------------------------
# Fichier provenant de TA caractérisation de dynamique
# ------------------------------------------------------------

DYNAMIC_SUMMARY_CSV = (
    Path.home()
    / "b200_dynamic_range_max_pulses"
    / "dynamic_range_max_summary.csv"
)

# Si ton fichier est ailleurs, change uniquement cette ligne.
#
# Exemple :
#
# DYNAMIC_SUMMARY_CSV = Path(
#     "/home/yousef/Documents/testing_scripts/GEVernova/"
#     "Fi_Wo_Ca_Dy/Dynamic/Dyn_Files/"
#     "b200_dynamic_range_max_pulses_BigRange/"
#     "dynamic_range_max_summary.csv"
# )


# ------------------------------------------------------------
# Bande utilisée
# ------------------------------------------------------------

F_MIN_HZ = 100e6
F_MAX_HZ = 2.0e9


# ------------------------------------------------------------
# Marges par rapport à Pmin et Pmax
# ------------------------------------------------------------

LOW_MARGIN_DB: float = 2.0
HIGH_MARGIN_DB: float = 2.0


# ------------------------------------------------------------
# Sécurité clipping
# ------------------------------------------------------------

CLIP_THRESHOLD: float = 1e-4


# ============================================================
# 2) STRUCTURES
# ============================================================

@dataclass
class DynamicLimits:
    frequency_hz: float
    gain_db: float

    pmin_dbm: float
    pmax_dbm: float

    lower_limit_dbm: float
    upper_limit_dbm: float

    dynamic_range_db: float


@dataclass
class GainDecision:
    current_gain_db: float
    next_gain_db: float | None

    measured_dbm: float

    pmin_dbm: float
    pmax_dbm: float

    lower_limit_dbm: float
    upper_limit_dbm: float

    reason: str

    needs_second_acquisition: bool


# ============================================================
# 3) OUTILS GAIN
# ============================================================

def _gain_index(
    gain_db: float,
) -> int:

    gains = np.asarray(
        GAINS_DB,
        dtype=float,
    )

    return int(
        np.argmin(
            np.abs(
                gains
                - float(gain_db)
            )
        )
    )


def nearest_gain(
    gain_db: float,
) -> float:

    return float(
        GAINS_DB[
            _gain_index(
                gain_db
            )
        ]
    )


def gain_up(
    gain_db: float,
) -> float | None:

    idx = _gain_index(
        gain_db
    )

    if idx >= len(GAINS_DB) - 1:
        return None

    return float(
        GAINS_DB[
            idx + 1
        ]
    )


def gain_down(
    gain_db: float,
) -> float | None:

    idx = _gain_index(
        gain_db
    )

    if idx <= 0:
        return None

    return float(
        GAINS_DB[
            idx - 1
        ]
    )


# ============================================================
# 4) TABLE DE DYNAMIQUE
# ============================================================

class DynamicRangeTable:
    """
    Charge dynamic_range_max_summary.csv.

    Colonnes utilisées :

        frequency_hz
        gain_db
        minimum_detectable_dbm
        maximum_linear_dbm

    Pour une fréquence intermédiaire, Pmin et Pmax
    sont interpolés linéairement.
    """

    def __init__(
        self,
        csv_file: Path,
    ) -> None:

        self.csv_file = Path(
            csv_file
        )

        if not self.csv_file.exists():

            raise FileNotFoundError(
                "Fichier de dynamique introuvable :\n"
                f"{self.csv_file}"
            )

        df = pd.read_csv(
            self.csv_file
        )

        required_columns = [
            "frequency_hz",
            "gain_db",
            "minimum_detectable_dbm",
            "maximum_linear_dbm",
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                "Colonnes absentes du fichier de dynamique : "
                f"{missing}"
            )

        # Conversion numérique
        for column in required_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        # On garde uniquement 100 MHz -> 2 GHz
        df = df[
            (
                df["frequency_hz"]
                >= F_MIN_HZ
            )
            &
            (
                df["frequency_hz"]
                <= F_MAX_HZ
            )
        ].copy()

        # Pmin et Pmax doivent exister
        df = df.dropna(
            subset=[
                "frequency_hz",
                "gain_db",
                "minimum_detectable_dbm",
                "maximum_linear_dbm",
            ]
        )

        if df.empty:

            raise ValueError(
                "Aucun point de dynamique valide "
                "entre 100 MHz et 2 GHz."
            )

        df = df.sort_values(
            [
                "gain_db",
                "frequency_hz",
            ]
        )

        self.df = df

        print()
        print(
            "Table de dynamique chargée :"
        )

        print(
            self.csv_file
        )

        print(
            f"Nombre de points : "
            f"{len(self.df)}"
        )

        print(
            "Gains disponibles :",
            sorted(
                self.df[
                    "gain_db"
                ].unique()
            ),
        )


    # ========================================================
    # Récupération Pmin/Pmax
    # ========================================================

    def get_limits(
        self,
        frequency_hz: float,
        gain_db: float,
    ) -> DynamicLimits:

        frequency_hz = float(
            frequency_hz
        )

        gain_db = nearest_gain(
            gain_db
        )

        gain_df = self.df[
            np.isclose(
                self.df["gain_db"],
                gain_db,
                atol=0.1,
            )
        ].copy()

        if gain_df.empty:

            raise ValueError(
                f"Aucune dynamique disponible "
                f"pour le gain {gain_db:.1f} dB."
            )

        gain_df = gain_df.sort_values(
            "frequency_hz"
        )

        frequencies = (
            gain_df[
                "frequency_hz"
            ].to_numpy(
                dtype=float
            )
        )

        pmins = (
            gain_df[
                "minimum_detectable_dbm"
            ].to_numpy(
                dtype=float
            )
        )

        pmaxs = (
            gain_df[
                "maximum_linear_dbm"
            ].to_numpy(
                dtype=float
            )
        )

        minimum_frequency = float(
            frequencies.min()
        )

        maximum_frequency = float(
            frequencies.max()
        )

        if not (
            minimum_frequency
            <= frequency_hz
            <= maximum_frequency
        ):

            raise ValueError(
                f"Fréquence "
                f"{frequency_hz / 1e6:.1f} MHz "
                f"hors plage de dynamique "
                f"pour gain {gain_db:.0f} dB : "
                f"[{minimum_frequency / 1e6:.1f}, "
                f"{maximum_frequency / 1e6:.1f}] MHz"
            )

        # ----------------------------------------------------
        # Interpolation
        # ----------------------------------------------------

        pmin_dbm = float(
            np.interp(
                frequency_hz,
                frequencies,
                pmins,
            )
        )

        pmax_dbm = float(
            np.interp(
                frequency_hz,
                frequencies,
                pmaxs,
            )
        )

        lower_limit_dbm = (
            pmin_dbm
            + LOW_MARGIN_DB
        )

        upper_limit_dbm = (
            pmax_dbm
            - HIGH_MARGIN_DB
        )

        return DynamicLimits(

            frequency_hz=
                frequency_hz,

            gain_db=
                gain_db,

            pmin_dbm=
                pmin_dbm,

            pmax_dbm=
                pmax_dbm,

            lower_limit_dbm=
                lower_limit_dbm,

            upper_limit_dbm=
                upper_limit_dbm,

            dynamic_range_db=
                pmax_dbm
                - pmin_dbm,
        )


# ============================================================
# 5) DÉCISION DU GAIN
# ============================================================

_DYNAMIC_TABLE: DynamicRangeTable | None = None


def get_dynamic_table() -> DynamicRangeTable:
    """Charge la table de dynamique une seule fois et la réutilise."""
    global _DYNAMIC_TABLE
    if _DYNAMIC_TABLE is None:
        _DYNAMIC_TABLE = DynamicRangeTable(DYNAMIC_SUMMARY_CSV)
    return _DYNAMIC_TABLE


def decide_next_gain(
    frequency_hz: float,
    current_gain_db: float,
    measured_dbm: float,
    clipping_fraction: float,
) -> GainDecision:
    """
    Compare la puissance mesurée à la dynamique
    caractérisée pour fréquence/gain.
    """

    dynamic_table = get_dynamic_table()

    current_gain_db = nearest_gain(
        current_gain_db
    )

    limits = dynamic_table.get_limits(
        frequency_hz,
        current_gain_db,
    )

    # ========================================================
    # CAS 1 : clipping
    # ========================================================

    if (
        clipping_fraction
        > CLIP_THRESHOLD
    ):

        new_gain = gain_down(
            current_gain_db
        )

        if new_gain is None:

            return GainDecision(
                current_gain_db=current_gain_db,
                next_gain_db=None,
                measured_dbm=measured_dbm,

                pmin_dbm=limits.pmin_dbm,
                pmax_dbm=limits.pmax_dbm,

                lower_limit_dbm=
                    limits.lower_limit_dbm,

                upper_limit_dbm=
                    limits.upper_limit_dbm,

                reason="CLIPPING mais gain déjà minimum",

                needs_second_acquisition=False,
            )

        return GainDecision(
            current_gain_db=current_gain_db,
            next_gain_db=new_gain,
            measured_dbm=measured_dbm,

            pmin_dbm=limits.pmin_dbm,
            pmax_dbm=limits.pmax_dbm,

            lower_limit_dbm=
                limits.lower_limit_dbm,

            upper_limit_dbm=
                limits.upper_limit_dbm,

            reason=(
                f"CLIPPING : "
                f"{current_gain_db:.0f}"
                f" -> "
                f"{new_gain:.0f} dB"
            ),

            needs_second_acquisition=True,
        )

    # ========================================================
    # CAS 2 : puissance trop forte
    # ========================================================

    if (
        measured_dbm
        >
        limits.upper_limit_dbm
    ):

        new_gain = gain_down(
            current_gain_db
        )

        if new_gain is None:

            return GainDecision(
                current_gain_db=current_gain_db,
                next_gain_db=None,
                measured_dbm=measured_dbm,

                pmin_dbm=limits.pmin_dbm,
                pmax_dbm=limits.pmax_dbm,

                lower_limit_dbm=
                    limits.lower_limit_dbm,

                upper_limit_dbm=
                    limits.upper_limit_dbm,

                reason="TROP_FORT@GAIN_MIN",

                needs_second_acquisition=False,
            )

        return GainDecision(
            current_gain_db=current_gain_db,
            next_gain_db=new_gain,
            measured_dbm=measured_dbm,

            pmin_dbm=limits.pmin_dbm,
            pmax_dbm=limits.pmax_dbm,

            lower_limit_dbm=
                limits.lower_limit_dbm,

            upper_limit_dbm=
                limits.upper_limit_dbm,

            reason=(
                f"TROP FORT : "
                f"{measured_dbm:.1f} dBm > "
                f"{limits.upper_limit_dbm:.1f} dBm "
                f"=> Gain "
                f"{current_gain_db:.0f}"
                f" -> "
                f"{new_gain:.0f} dB"
            ),

            needs_second_acquisition=True,
        )

    # ========================================================
    # CAS 3 : puissance trop faible
    # ========================================================

    if (
        measured_dbm
        <
        limits.lower_limit_dbm
    ):

        new_gain = gain_up(
            current_gain_db
        )

        if new_gain is None:

            return GainDecision(
                current_gain_db=current_gain_db,
                next_gain_db=None,
                measured_dbm=measured_dbm,

                pmin_dbm=limits.pmin_dbm,
                pmax_dbm=limits.pmax_dbm,

                lower_limit_dbm=
                    limits.lower_limit_dbm,

                upper_limit_dbm=
                    limits.upper_limit_dbm,

                reason="TROP_FAIBLE@GAIN_MAX",

                needs_second_acquisition=False,
            )

        return GainDecision(
            current_gain_db=current_gain_db,
            next_gain_db=new_gain,
            measured_dbm=measured_dbm,

            pmin_dbm=limits.pmin_dbm,
            pmax_dbm=limits.pmax_dbm,

            lower_limit_dbm=
                limits.lower_limit_dbm,

            upper_limit_dbm=
                limits.upper_limit_dbm,

            reason=(
                f"TROP FAIBLE : "
                f"{measured_dbm:.1f} dBm < "
                f"{limits.lower_limit_dbm:.1f} dBm "
                f"=> Gain "
                f"{current_gain_db:.0f}"
                f" -> "
                f"{new_gain:.0f} dB"
            ),

            needs_second_acquisition=True,
        )

    # ========================================================
    # CAS 4 : gain adapté
    # ========================================================

    return GainDecision(
        current_gain_db=current_gain_db,
        next_gain_db=None,
        measured_dbm=measured_dbm,

        pmin_dbm=limits.pmin_dbm,
        pmax_dbm=limits.pmax_dbm,

        lower_limit_dbm=
            limits.lower_limit_dbm,

        upper_limit_dbm=
            limits.upper_limit_dbm,

        reason="GAIN_OK",

        needs_second_acquisition=False,
    )


# ============================================================
# 6) GAIN MAP
# ============================================================

class GainMap:
    """
    Stocke le gain finalement utilisé
    pour chaque fréquence.
    """

    def __init__(
        self,
    ) -> None:

        self._data: dict[
            float,
            float,
        ] = {}


    def set(
        self,
        frequency_hz: float,
        gain_db: float,
    ) -> None:

        self._data[
            float(frequency_hz)
        ] = float(
            gain_db
        )


    def get(
        self,
        frequency_hz: float,
    ) -> float | None:

        return self._data.get(
            float(
                frequency_hz
            )
        )


    def frequencies_hz(
        self,
    ) -> list[float]:

        return sorted(
            self._data.keys()
        )


    def gains_db(
        self,
    ) -> list[float]:

        return [
            self._data[f]
            for f
            in self.frequencies_hz()
        ]


    def as_arrays(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        frequencies = np.asarray(
            self.frequencies_hz(),
            dtype=float,
        )

        gains = np.asarray(
            self.gains_db(),
            dtype=float,
        )

        return (
            frequencies,
            gains,
        )


    def clear(
        self,
    ) -> None:

        self._data.clear()


# ============================================================
# 7) TEST DU MODULE
# ============================================================

if __name__ == "__main__":

    table = get_dynamic_table()

    print()
    print(
        "Exemple à 1 GHz / gain 40 dB :"
    )

    limits = table.get_limits(
        1.0e9,
        40.0,
    )

    print(
        f"Pmin = "
        f"{limits.pmin_dbm:.2f} dBm"
    )

    print(
        f"Pmax = "
        f"{limits.pmax_dbm:.2f} dBm"
    )

    print(
        f"Zone avec marges = "
        f"[{limits.lower_limit_dbm:.2f} ; "
        f"{limits.upper_limit_dbm:.2f}] dBm"
    )