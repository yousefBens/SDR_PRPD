#!/usr/bin/env python3

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1) CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Dossier contenant tes mesures de dynamique
# ------------------------------------------------------------

INPUT_DIR = Path(
    "/home/yousef/Documents/testing_scripts/GEVernova/"
    "Fi_Wo_Ca_Dy/Dynamic/Dyn_Files/"
    "b200_dynamic_range_max_pulses_BigRange"
)


# ------------------------------------------------------------
# CHOISIS ICI LA TABLE DE CALIBRATION DE SORTIE
# ------------------------------------------------------------
# /home/yousef/Documents/testing_scripts/GEVernova/Fi_Wo_Ca_Dy/Calib/Calib_with_dynam/Final_Calib

OUTPUT_PICKLE = Path(
    "/home/yousef/Documents/testing_scripts/GEVernova/"
    "Fi_Wo_Ca_Dy/Calib/Calib_with_dynam/Final_Calib/"
    "rx2_calibration_from_dynamic.pickle"
)


# CSV de diagnostic généré automatiquement
OUTPUT_CSV = OUTPUT_PICKLE.with_suffix(".csv")

# Modèles affines complets pour analyse
OUTPUT_MODELS_PICKLE = OUTPUT_PICKLE.with_name(
    OUTPUT_PICKLE.stem + "_models.pickle"
)

# Graphes
PLOTS_DIR = OUTPUT_PICKLE.parent / (
    OUTPUT_PICKLE.stem + "_plots"
)


# ============================================================
# 2) FICHIERS D'ENTRÉE
# ============================================================

POINTS_CSV = (
    INPUT_DIR
    / "dynamic_range_max_aggregated_points.csv"
)

SUMMARY_CSV = (
    INPUT_DIR
    / "dynamic_range_max_summary.csv"
)

RAW_CSV = (
    INPUT_DIR
    / "dynamic_range_max_raw_measurements.csv"
)


# ============================================================
# 3) PARAMÈTRES DE QUALITÉ
# ============================================================

# Nombre minimum de points linéaires pour faire une calibration
MIN_POINTS = 4

# Le signal doit être suffisamment au-dessus du bruit
MIN_EXCESS_DB = 12.0

# Probabilité de détection minimale
MIN_DETECTION_PROBABILITY = 0.90

# Compression maximale autorisée
MAX_COMPRESSION_DB = 1.0

# Eviter la proximité du clipping / saturation numérique
MAX_DBFS_FOR_CALIBRATION = -0.5

# Fraction maximale d'échantillons proches du clipping
MAX_CLIPPING_FRACTION = 1e-4


# ------------------------------------------------------------
# Vérification de la régression
# ------------------------------------------------------------

# Dans une bonne zone linéaire :
#
# max_dbfs = a * Pin + b
#
# La pente doit être proche de 1 dB/dB.
MAX_SLOPE_ERROR = 0.15

MIN_R2 = 0.995

# RMSE maximum de la calibration en dB
MAX_RMSE_DB = 1.5


# ------------------------------------------------------------
# Rejet itératif des valeurs aberrantes
# ------------------------------------------------------------

OUTLIER_SIGMA = 3.5

MAX_OUTLIER_ITERATIONS = 5


# ============================================================
# 4) CHARGEMENT
# ============================================================

def load_data():

    if not POINTS_CSV.exists():
        raise FileNotFoundError(
            f"Fichier introuvable :\n{POINTS_CSV}"
        )

    points = pd.read_csv(POINTS_CSV)

    print()
    print("=" * 75)
    print("CHARGEMENT DE LA BASE DE DYNAMIQUE")
    print("=" * 75)

    print(f"Points : {POINTS_CSV}")
    print(f"Nombre de lignes : {len(points)}")

    required = [
        "frequency_hz",
        "frequency_mhz",
        "gain_db",
        "input_rx2_dbm",
        "max_dbfs",
        "median_max_excess_db",
        "detection_probability",
        "clipping_fraction",
        "compression_db",
        "linear",
    ]

    missing = [
        c for c in required
        if c not in points.columns
    ]

    if missing:
        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(missing)
        )

    # Conversion numérique
    numeric_columns = [
        "frequency_hz",
        "frequency_mhz",
        "gain_db",
        "input_rx2_dbm",
        "max_dbfs",
        "median_max_excess_db",
        "detection_probability",
        "clipping_fraction",
        "compression_db",
    ]

    for c in numeric_columns:

        points[c] = pd.to_numeric(
            points[c],
            errors="coerce"
        )

    # Conversion robuste du booléen
    if points["linear"].dtype != bool:

        points["linear"] = (
            points["linear"]
            .astype(str)
            .str.lower()
            .isin([
                "true",
                "1",
                "yes"
            ])
        )

    return points


# ============================================================
# 5) SÉLECTION DES POINTS FIABLES
# ============================================================

def select_calibration_points(df):

    """
    Ne garde que les points réellement intéressants
    pour construire la calibration.
    """

    valid = df.copy()

    # --------------------------------------------------------
    # Point déclaré linéaire par ta caractérisation
    # --------------------------------------------------------

    mask = valid["linear"].copy()

    # --------------------------------------------------------
    # Signal suffisamment au-dessus du noise floor
    # --------------------------------------------------------

    mask &= (
        valid["median_max_excess_db"]
        >= MIN_EXCESS_DB
    )

    # --------------------------------------------------------
    # Détection fiable
    # --------------------------------------------------------

    mask &= (
        valid["detection_probability"]
        >= MIN_DETECTION_PROBABILITY
    )

    # --------------------------------------------------------
    # Pas de clipping
    # --------------------------------------------------------

    mask &= (
        valid["clipping_fraction"]
        <= MAX_CLIPPING_FRACTION
    )

    # --------------------------------------------------------
    # Pas trop proche du full-scale
    # --------------------------------------------------------

    mask &= (
        valid["max_dbfs"]
        <= MAX_DBFS_FOR_CALIBRATION
    )

    # --------------------------------------------------------
    # Compression faible
    # --------------------------------------------------------

    # compression_db peut être NaN sur certains points.
    compression_ok = (
        valid["compression_db"].isna()
        |
        (
            np.abs(valid["compression_db"])
            <= MAX_COMPRESSION_DB
        )
    )

    mask &= compression_ok

    valid = valid[mask].copy()

    valid = valid.dropna(
        subset=[
            "input_rx2_dbm",
            "max_dbfs"
        ]
    )

    valid = valid.sort_values(
        "input_rx2_dbm"
    )

    return valid


# ============================================================
# 6) RÉGRESSION ROBUSTE
# ============================================================

def robust_linear_fit(x, y):

    """
    Fit :

        y = a*x + b

    ici :

        x = max_dbfs
        y = Pin réelle en dBm

    Rejet itératif des outliers.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    for _ in range(MAX_OUTLIER_ITERATIONS):

        xx = x[mask]
        yy = y[mask]

        if len(xx) < MIN_POINTS:
            break

        a, b = np.polyfit(
            xx,
            yy,
            1
        )

        prediction = (
            a * xx
            + b
        )

        residuals = (
            yy - prediction
        )

        median = np.median(residuals)

        mad = np.median(
            np.abs(
                residuals - median
            )
        )

        sigma = (
            1.4826 * mad
        )

        # Si dispersion pratiquement nulle
        if sigma < 1e-9:
            break

        local_good = (
            np.abs(
                residuals - median
            )
            <= OUTLIER_SIGMA * sigma
        )

        old_indices = np.where(mask)[0]

        new_mask = np.zeros_like(
            mask,
            dtype=bool
        )

        new_mask[
            old_indices[local_good]
        ] = True

        if np.array_equal(
            new_mask,
            mask
        ):
            break

        mask = new_mask

    xx = x[mask]
    yy = y[mask]

    if len(xx) < 2:

        return None

    a, b = np.polyfit(
        xx,
        yy,
        1
    )

    prediction = (
        a * xx
        + b
    )

    residuals = (
        yy - prediction
    )

    ss_res = np.sum(
        residuals ** 2
    )

    ss_tot = np.sum(
        (
            yy - np.mean(yy)
        ) ** 2
    )

    if ss_tot > 0:

        r2 = (
            1.0
            - ss_res / ss_tot
        )

    else:

        r2 = 1.0

    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    max_error = float(
        np.max(
            np.abs(residuals)
        )
    )

    return {
        "slope": float(a),
        "intercept": float(b),
        "r2": float(r2),
        "rmse_db": rmse,
        "max_error_db": max_error,
        "mask": mask,
        "x": xx,
        "y": yy,
    }


# ============================================================
# 7) CALCUL DE LA RÉFÉRENCE OFFSET
# ============================================================

def calculate_reference(valid_df):

    """
    Modèle utilisé par ton application :

        Pin_dBm = max_dbfs + reference_dbm

    donc :

        reference_dbm = Pin_dBm - max_dbfs


    On utilise la médiane sur plusieurs points linéaires.
    """

    refs = (
        valid_df["input_rx2_dbm"].to_numpy(dtype=float)
        -
        valid_df["max_dbfs"].to_numpy(dtype=float)
    )

    reference = float(
        np.median(refs)
    )

    # Résultat avec calibration offset
    estimated = (
        valid_df["max_dbfs"].to_numpy(dtype=float)
        +
        reference
    )

    actual = (
        valid_df["input_rx2_dbm"].to_numpy(dtype=float)
    )

    errors = (
        estimated - actual
    )

    rmse = float(
        np.sqrt(
            np.mean(errors ** 2)
        )
    )

    mean_error = float(
        np.mean(errors)
    )

    max_error = float(
        np.max(
            np.abs(errors)
        )
    )

    std_error = float(
        np.std(errors)
    )

    return {
        "reference_dbm": reference,
        "rmse_db": rmse,
        "mean_error_db": mean_error,
        "std_error_db": std_error,
        "max_error_db": max_error,
    }


# ============================================================
# 8) CONSTRUCTION DE LA CALIBRATION
# ============================================================

def build_calibration(points):

    calibration = {
        0: {
            "RX2": {}
        }
    }

    models = {
        0: {
            "RX2": {}
        }
    }

    summary_rows = []

    grouped = points.groupby(
        [
            "frequency_hz",
            "gain_db"
        ],
        sort=True
    )

    total_groups = len(grouped)

    print()
    print("=" * 75)
    print("CONSTRUCTION DE LA TABLE DE CALIBRATION")
    print("=" * 75)

    print(
        f"Nombre de couples fréquence/gain : "
        f"{total_groups}"
    )

    valid_count = 0

    for (
        frequency_hz,
        gain_db
    ), group in grouped:

        frequency_hz = float(
            frequency_hz
        )

        gain_db = float(
            gain_db
        )

        selected = select_calibration_points(
            group
        )

        n_total = len(group)
        n_selected = len(selected)

        # Valeurs par défaut
        status = "INVALID"
        reference_dbm = np.nan

        slope = np.nan
        affine_intercept = np.nan

        r2 = np.nan
        affine_rmse = np.nan

        offset_rmse = np.nan
        offset_max_error = np.nan
        offset_mean_error = np.nan
        offset_std_error = np.nan

        pin_min = np.nan
        pin_max = np.nan

        dbfs_min = np.nan
        dbfs_max = np.nan

        if n_selected >= MIN_POINTS:

            fit = robust_linear_fit(
                selected["max_dbfs"],
                selected["input_rx2_dbm"]
            )

            if fit is not None:

                # ------------------------------------------------
                # Garder uniquement les points après rejet outliers
                # ------------------------------------------------

                robust_mask = fit["mask"]

                selected_robust = selected.iloc[
                    np.where(robust_mask)[0]
                ].copy()

                n_selected = len(
                    selected_robust
                )

                slope = fit["slope"]
                affine_intercept = fit[
                    "intercept"
                ]

                r2 = fit["r2"]

                affine_rmse = fit[
                    "rmse_db"
                ]

                if n_selected >= MIN_POINTS:

                    offset = calculate_reference(
                        selected_robust
                    )

                    reference_dbm = (
                        offset[
                            "reference_dbm"
                        ]
                    )

                    offset_rmse = (
                        offset[
                            "rmse_db"
                        ]
                    )

                    offset_max_error = (
                        offset[
                            "max_error_db"
                        ]
                    )

                    offset_mean_error = (
                        offset[
                            "mean_error_db"
                        ]
                    )

                    offset_std_error = (
                        offset[
                            "std_error_db"
                        ]
                    )

                    pin_min = float(
                        selected_robust[
                            "input_rx2_dbm"
                        ].min()
                    )

                    pin_max = float(
                        selected_robust[
                            "input_rx2_dbm"
                        ].max()
                    )

                    dbfs_min = float(
                        selected_robust[
                            "max_dbfs"
                        ].min()
                    )

                    dbfs_max = float(
                        selected_robust[
                            "max_dbfs"
                        ].max()
                    )

                    # ============================================
                    # VALIDATION
                    # ============================================

                    slope_ok = (
                        abs(
                            slope - 1.0
                        )
                        <= MAX_SLOPE_ERROR
                    )

                    r2_ok = (
                        r2 >= MIN_R2
                    )

                    rmse_ok = (
                        offset_rmse
                        <= MAX_RMSE_DB
                    )

                    if (
                        slope_ok
                        and r2_ok
                        and rmse_ok
                    ):

                        status = "VALID"

                        valid_count += 1

                        # ========================================
                        # TABLE COMPATIBLE AVEC TON ANCIEN CODE
                        # ========================================

                        if (
                            frequency_hz
                            not in calibration[0]["RX2"]
                        ):

                            calibration[
                                0
                            ][
                                "RX2"
                            ][
                                frequency_hz
                            ] = {}

                        calibration[
                            0
                        ][
                            "RX2"
                        ][
                            frequency_hz
                        ][
                            gain_db
                        ] = reference_dbm

                        # ========================================
                        # MODELE COMPLET
                        # ========================================

                        if (
                            frequency_hz
                            not in models[0]["RX2"]
                        ):

                            models[
                                0
                            ][
                                "RX2"
                            ][
                                frequency_hz
                            ] = {}

                        models[
                            0
                        ][
                            "RX2"
                        ][
                            frequency_hz
                        ][
                            gain_db
                        ] = {
                            "reference_dbm":
                                reference_dbm,

                            "slope":
                                slope,

                            "intercept":
                                affine_intercept,

                            "r2":
                                r2,

                            "offset_rmse_db":
                                offset_rmse,

                            "offset_max_error_db":
                                offset_max_error,

                            "input_min_dbm":
                                pin_min,

                            "input_max_dbm":
                                pin_max,

                            "dbfs_min":
                                dbfs_min,

                            "dbfs_max":
                                dbfs_max,

                            "n_points":
                                n_selected,
                        }

        summary_rows.append(
            {
                "frequency_hz":
                    frequency_hz,

                "frequency_mhz":
                    frequency_hz / 1e6,

                "gain_db":
                    gain_db,

                "status":
                    status,

                "n_total_points":
                    n_total,

                "n_calibration_points":
                    n_selected,

                # Calibration utilisée
                "reference_dbm":
                    reference_dbm,

                # Modèle affine de contrôle
                "affine_slope":
                    slope,

                "affine_intercept_dbm":
                    affine_intercept,

                "r2":
                    r2,

                "affine_rmse_db":
                    affine_rmse,

                # Qualité du modèle OFFSET réellement utilisé
                "offset_rmse_db":
                    offset_rmse,

                "offset_mean_error_db":
                    offset_mean_error,

                "offset_std_error_db":
                    offset_std_error,

                "offset_max_error_db":
                    offset_max_error,

                # Plage utilisée
                "input_min_dbm":
                    pin_min,

                "input_max_dbm":
                    pin_max,

                "dbfs_min":
                    dbfs_min,

                "dbfs_max":
                    dbfs_max,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print(
        f"Calibrations VALIDES : "
        f"{valid_count}/{total_groups}"
    )

    return (
        calibration,
        models,
        summary
    )


# ============================================================
# 9) GRAPHE DE CONTRÔLE
# ============================================================

def make_validation_plots(
    points,
    summary
):

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    valid_summary = summary[
        summary["status"] == "VALID"
    ].copy()

    for _, row in valid_summary.iterrows():

        freq_hz = float(
            row["frequency_hz"]
        )

        gain_db = float(
            row["gain_db"]
        )

        reference = float(
            row["reference_dbm"]
        )

        df = points[
            np.isclose(
                points["frequency_hz"],
                freq_hz
            )
            &
            np.isclose(
                points["gain_db"],
                gain_db
            )
        ].copy()

        df = df.sort_values(
            "input_rx2_dbm"
        )

        calibrated_dbm = (
            df["max_dbfs"].to_numpy(dtype=float)
            +
            reference
        )

        pin = df[
            "input_rx2_dbm"
        ].to_numpy(dtype=float)

        plt.figure(
            figsize=(9, 7)
        )

        # Toutes les mesures
        plt.plot(
            pin,
            calibrated_dbm,
            "o-",
            label="SDR après calibration"
        )

        # Idéal
        pmin = min(
            pin.min(),
            calibrated_dbm.min()
        )

        pmax = max(
            pin.max(),
            calibrated_dbm.max()
        )

        plt.plot(
            [pmin, pmax],
            [pmin, pmax],
            "--",
            label="Idéal : Pmes = Pin"
        )

        # Zone réellement utilisée pour la calibration
        plt.axvspan(
            row["input_min_dbm"],
            row["input_max_dbm"],
            alpha=0.15,
            label="Zone utilisée pour calibration"
        )

        plt.xlabel(
            "Puissance réellement injectée RX2 (dBm)"
        )

        plt.ylabel(
            "Puissance SDR calibrée (dBm)"
        )

        plt.title(
            f"Validation calibration\n"
            f"{freq_hz / 1e6:.1f} MHz | "
            f"Gain {gain_db:.0f} dB\n"
            f"RMSE = {row['offset_rmse_db']:.2f} dB | "
            f"R² = {row['r2']:.5f}"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()

        filename = (
            PLOTS_DIR
            / (
                f"cal_{freq_hz / 1e6:.0f}MHz_"
                f"gain_{gain_db:.0f}dB.png"
            )
        )

        plt.savefig(
            filename,
            dpi=150
        )

        plt.close()


# ============================================================
# 10) SAUVEGARDE
# ============================================================

def save_results(
    calibration,
    models,
    summary
):

    OUTPUT_PICKLE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Table compatible avec :
    #
    # calibration[0]["RX2"][freq][gain]
    # --------------------------------------------------------

    with OUTPUT_PICKLE.open(
        "wb"
    ) as f:

        pickle.dump(
            calibration,
            f
        )

    # --------------------------------------------------------
    # Modèles + métadonnées
    # --------------------------------------------------------

    with OUTPUT_MODELS_PICKLE.open(
        "wb"
    ) as f:

        pickle.dump(
            models,
            f
        )

    # --------------------------------------------------------
    # CSV lisible
    # --------------------------------------------------------

    summary.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("=" * 75)
    print("FICHIERS GÉNÉRÉS")
    print("=" * 75)

    print(
        "Table calibration :"
    )
    print(
        OUTPUT_PICKLE
    )

    print()
    print(
        "Diagnostic CSV :"
    )
    print(
        OUTPUT_CSV
    )

    print()
    print(
        "Modèles complets :"
    )
    print(
        OUTPUT_MODELS_PICKLE
    )

    print()
    print(
        "Graphes :"
    )
    print(
        PLOTS_DIR
    )


# ============================================================
# 11) TEST DE LA TABLE
# ============================================================

def test_calibration(
    calibration,
    dbfs,
    frequency_hz,
    gain_db
):

    table = calibration[
        0
    ][
        "RX2"
    ]

    freqs = np.array(
        sorted(
            float(f)
            for f in table.keys()
        ),
        dtype=float
    )

    # Référence pour chaque fréquence au gain demandé
    cal_freqs = []
    cal_refs = []

    for f in freqs:

        gains_table = table[f]

        available_gains = np.array(
            [
                float(g)
                for g in gains_table.keys()
            ],
            dtype=float
        )

        if len(
            available_gains
        ) == 0:
            continue

        selected_gain = (
            available_gains[
                np.argmin(
                    np.abs(
                        available_gains
                        - gain_db
                    )
                )
            ]
        )

        # Evite de prendre un gain complètement différent
        if (
            abs(
                selected_gain
                - gain_db
            )
            > 1.0
        ):
            continue

        cal_freqs.append(
            f
        )

        cal_refs.append(
            gains_table[
                selected_gain
            ]
        )

    if len(cal_freqs) == 0:

        raise ValueError(
            f"Aucune calibration disponible "
            f"pour gain {gain_db} dB"
        )

    cal_freqs = np.array(
        cal_freqs,
        dtype=float
    )

    cal_refs = np.array(
        cal_refs,
        dtype=float
    )

    reference = float(
        np.interp(
            frequency_hz,
            cal_freqs,
            cal_refs
        )
    )

    dbm = (
        float(dbfs)
        +
        reference
    )

    return dbm


# ============================================================
# MAIN
# ============================================================

def main():

    points = load_data()

    calibration, models, summary = (
        build_calibration(
            points
        )
    )

    save_results(
        calibration,
        models,
        summary
    )

    print()
    print("=" * 75)
    print("RÉSUMÉ QUALITÉ")
    print("=" * 75)

    valid = summary[
        summary["status"] == "VALID"
    ].copy()

    if valid.empty:

        print(
            "ATTENTION : aucune calibration "
            "n'a passé les critères."
        )

        return

    print(
        valid[
            [
                "frequency_mhz",
                "gain_db",
                "reference_dbm",
                "affine_slope",
                "r2",
                "offset_rmse_db",
                "offset_max_error_db",
                "input_min_dbm",
                "input_max_dbm",
            ]
        ].to_string(
            index=False
        )
    )

    # Décommente si tu veux créer tous les graphes
    # make_validation_plots(
    #     points,
    #     summary
    # )


if __name__ == "__main__":
    main()