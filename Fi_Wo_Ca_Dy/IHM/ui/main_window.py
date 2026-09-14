#!/usr/bin/env python3
"""
ui/main_window.py
-----------------
Fenêtre principale PyQt6 de l'IHM SDR.

Fonctions :
  - Scan spectral
  - Acquisition PRPD
  - Conservation des IQ bruts utilisés pour le PRPD
  - Export complet :
      spectrum.csv
      prpd.csv
      prpd_matrix_256x256.npz
      iq_raw.npz
      metadata.json
"""

from __future__ import annotations

import csv
import json
import time

from datetime import datetime
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.calibration import (
    build_all_gain_calibrations,
    load_power_calibration,
)

from core.usrp_backend import (
    FreqResult,
    PrpdThread,
    ScanThread,
)

from ui.config_panel import ConfigPanel
from ui.prpd_panel import PrpdPanel
from ui.spectrum_panel import SpectrumPanel


class MainWindow(QMainWindow):
    """Fenêtre principale IHM SDR."""

    APP_VERSION = "1.0.0"

    # ==========================================================
    # INITIALISATION
    # ==========================================================

    def __init__(self) -> None:
        super().__init__()

        # ------------------------------------------------------
        # Threads
        # ------------------------------------------------------

        self._scan_thread = None
        self._prpd_thread = None

        # ------------------------------------------------------
        # Résultats scan spectral
        # ------------------------------------------------------

        self._scan_results: list[FreqResult] = []

        self._scan_start_time: float = 0.0

        # ------------------------------------------------------
        # Calibration
        # ------------------------------------------------------

        self._cal_table: dict | None = None

        # {gain_db: calibration}
        self._all_cal: dict = {}

        # ------------------------------------------------------
        # Dernières données PRPD
        # ------------------------------------------------------

        self._last_prpd_phases = np.array(
            [],
            dtype=float,
        )

        self._last_prpd_amps = np.array(
            [],
            dtype=float,
        )

        self._last_prpd_info: dict = {}

        # ------------------------------------------------------
        # Spectre éventuellement associé au PRPD
        # ------------------------------------------------------

        self._last_prpd_spec_freqs = np.array(
            [],
            dtype=float,
        )

        self._last_prpd_spec_dbfs = np.array(
            [],
            dtype=float,
        )

        # ------------------------------------------------------
        # IQ BRUT provenant du PRPD
        # ------------------------------------------------------

        self._last_iq_samples = np.array(
            [],
            dtype=np.complex64,
        )

        # ------------------------------------------------------
        # Paramètres utilisés lors du dernier PRPD
        # ------------------------------------------------------

        self._last_prpd_params: dict = {}

        # ------------------------------------------------------
        # Fenêtre
        # ------------------------------------------------------

        self.setWindowTitle(
            f"GE IHM — Détection Décharges Partielles SDR "
            f"v{self.APP_VERSION}"
        )

        self.setMinimumSize(
            1300,
            780,
        )

        self.resize(
            1520,
            880,
        )

        # ------------------------------------------------------

        self._build_ui()
        self._connect_signals()
        self._load_calibration()

    # ==========================================================
    # CONSTRUCTION UI
    # ==========================================================

    def _build_ui(self) -> None:

        central = QWidget()

        self.setCentralWidget(
            central
        )

        # ------------------------------------------------------
        # Splitter principal
        # ------------------------------------------------------

        splitter = QSplitter(
            Qt.Orientation.Horizontal
        )

        # ------------------------------------------------------
        # Panneau configuration
        # ------------------------------------------------------

        self.config_panel = ConfigPanel()

        splitter.addWidget(
            self.config_panel
        )

        # ------------------------------------------------------
        # Zone droite
        # ------------------------------------------------------

        right = QWidget()

        right_lay = QVBoxLayout(
            right
        )

        right_lay.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        right_lay.setSpacing(
            0
        )

        # ------------------------------------------------------
        # Onglets
        # ------------------------------------------------------

        self.tabs = QTabWidget()

        self.tabs.setObjectName(
            "MainTabs"
        )

        # ------------------------------------------------------
        # Spectre
        # ------------------------------------------------------

        self.spectrum_panel = SpectrumPanel()

        self.tabs.addTab(
            self.spectrum_panel,
            "Spectre dBm",
        )

        # ------------------------------------------------------
        # PRPD
        # ------------------------------------------------------

        self.prpd_panel = PrpdPanel()

        self.tabs.addTab(
            self.prpd_panel,
            "PRPD",
        )

        # ------------------------------------------------------
        # Journal
        # ------------------------------------------------------

        self.log_edit = QTextEdit()

        self.log_edit.setReadOnly(
            True
        )

        self.log_edit.setObjectName(
            "LogEdit"
        )

        self.tabs.addTab(
            self.log_edit,
            "Journal",
        )

        right_lay.addWidget(
            self.tabs
        )

        # ------------------------------------------------------
        # Barre progression
        # ------------------------------------------------------

        self.progress_bar = QProgressBar()

        self.progress_bar.setRange(
            0,
            100,
        )

        self.progress_bar.setValue(
            0
        )

        self.progress_bar.setVisible(
            False
        )

        self.progress_bar.setFixedHeight(
            16
        )

        right_lay.addWidget(
            self.progress_bar
        )

        # ------------------------------------------------------

        splitter.addWidget(
            right
        )

        splitter.setStretchFactor(
            0,
            0,
        )

        splitter.setStretchFactor(
            1,
            1,
        )

        # ------------------------------------------------------
        # Layout principal
        # ------------------------------------------------------

        main_lay = QVBoxLayout(
            central
        )

        main_lay.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        main_lay.addWidget(
            splitter
        )

        # ------------------------------------------------------

        self._setup_status_bar()

    # ==========================================================
    # STATUS BAR
    # ==========================================================

    def _setup_status_bar(self) -> None:

        sb = QStatusBar()

        self.setStatusBar(
            sb
        )

        self.status_lbl_main = QLabel(
            "Prêt"
        )

        self.status_lbl_freq = QLabel(
            ""
        )

        self.status_lbl_cal = QLabel(
            "Calibration : non chargée"
        )

        sb.addWidget(
            self.status_lbl_main
        )

        sb.addPermanentWidget(
            self.status_lbl_freq
        )

        sb.addPermanentWidget(
            self.status_lbl_cal
        )

    # ==========================================================
    # CONNEXIONS
    # ==========================================================

    def _connect_signals(self) -> None:

        self.config_panel.start_scan.connect(
            self._on_start_scan
        )

        self.config_panel.stop_scan.connect(
            self._on_stop_scan
        )

        self.config_panel.start_prpd.connect(
            self._on_start_prpd
        )

        self.config_panel.stop_prpd.connect(
            self._on_stop_prpd
        )

        self.config_panel.export_results.connect(
            self._on_export
        )

        self.spectrum_panel.freq_selected.connect(
            self._on_freq_selected
        )

    # ==========================================================
    # CALIBRATION
    # ==========================================================

    def _load_calibration(self) -> None:

        try:

            self._cal_table = (
                load_power_calibration()
            )

            self._all_cal = (
                build_all_gain_calibrations(
                    self._cal_table
                )
            )

            gains_str = ", ".join(
                f"{g:.0f}"
                for g in sorted(
                    self._all_cal.keys()
                )
            )

            self.status_lbl_cal.setText(
                f"Calibration : [{gains_str}] dB"
            )

            self._log(
                "Calibration chargée — "
                f"gains disponibles : "
                f"{gains_str} dB"
            )

        except FileNotFoundError as exc:

            self.status_lbl_cal.setText(
                "Calibration : fichier introuvable"
            )

            self._log(
                f"[WARN] Calibration non disponible : "
                f"{exc}"
            )

            QMessageBox.warning(
                self,
                "Calibration manquante",
                f"Fichier de calibration non trouvé :\n"
                f"{exc}\n\n"
                f"Vérifiez l'emplacement du fichier "
                f"pickle de calibration.",
            )

        except Exception as exc:

            self.status_lbl_cal.setText(
                "Calibration : erreur"
            )

            self._log(
                f"[ERROR] Calibration : {exc}"
            )

    # ==========================================================

    def _get_cal_for_gain(
        self,
        gain_db: float,
    ) -> dict | None:

        if not self._all_cal:
            return None

        if gain_db in self._all_cal:
            return self._all_cal[
                gain_db
            ]

        arr = np.asarray(
            sorted(
                self._all_cal.keys()
            )
        )

        nearest = float(
            arr[
                np.argmin(
                    np.abs(
                        arr
                        - gain_db
                    )
                )
            ]
        )

        return self._all_cal.get(
            nearest
        )

    # ==========================================================
    # SCAN SPECTRAL
    # ==========================================================

    def _on_start_scan(self) -> None:

        params = (
            self.config_panel.get_params()
        )

        if not self._all_cal:

            QMessageBox.critical(
                self,
                "Calibration manquante",
                "Impossible de lancer le scan "
                "sans calibration.\n"
                "Vérifiez le fichier pickle "
                "de calibration.",
            )

            return

        # ------------------------------------------------------

        self._scan_results.clear()

        self.spectrum_panel.clear()

        self.progress_bar.setValue(
            0
        )

        self.progress_bar.setVisible(
            True
        )

        self.config_panel.set_scanning(
            True
        )

        self._scan_start_time = (
            time.perf_counter()
        )

        # ------------------------------------------------------

        n_freqs = int(
            (
                params["f_stop_hz"]
                - params["f_start_hz"]
            )
            / params["step_hz"]
        ) + 1

        self.status_lbl_main.setText(
            f"Scan en cours — "
            f"{n_freqs} fréquences | "
            f"USRP B200"
        )

        self._log(
            f"\n{'='*55}\n"
            f"Démarrage scan USRP\n"
            f"F : "
            f"{params['f_start_hz']/1e6:.0f}"
            f" → "
            f"{params['f_stop_hz']/1e6:.0f} MHz | "
            f"Pas "
            f"{params['step_hz']/1e6:.0f} MHz | "
            f"{n_freqs} points\n"
            f"{'='*55}"
        )

        # ------------------------------------------------------

        self._scan_thread = ScanThread(
            params,
            self._all_cal,
        )

        t = self._scan_thread

        t.result_ready.connect(
            self._on_scan_result
        )

        t.progress.connect(
            self.progress_bar.setValue
        )

        t.scan_done.connect(
            self._on_scan_done
        )

        t.log_message.connect(
            self._log
        )

        t.error.connect(
            self._on_error
        )

        t.start()

    # ==========================================================

    def _on_stop_scan(self) -> None:

        if (
            self._scan_thread
            and self._scan_thread.isRunning()
        ):

            self._scan_thread.stop()

            self.status_lbl_main.setText(
                "Scan interrompu."
            )

    # ==========================================================

    def _on_scan_result(
        self,
        result: FreqResult,
    ) -> None:

        self._scan_results.append(
            result
        )

        self.spectrum_panel.add_result(
            result
        )

        self.status_lbl_freq.setText(
            f"{result.freq_mhz:.1f} MHz | "
            f"G={result.gain_db:.0f} dB | "
            f"{result.max_dbm:.1f} dBm"
        )

    # ==========================================================

    def _on_scan_done(
        self,
        results: list[FreqResult],
    ) -> None:

        self._scan_results = results

        elapsed = (
            time.perf_counter()
            - self._scan_start_time
        )

        n = len(
            results
        )

        self.progress_bar.setValue(
            100
        )

        self.config_panel.set_scanning(
            False
        )

        self.status_lbl_main.setText(
            f"Scan terminé — "
            f"{n} points en "
            f"{elapsed:.1f} s"
        )

        if n > 0:

            self._log(
                f"\nScan terminé : "
                f"{n} points | "
                f"{elapsed:.1f} s | "
                f"{elapsed/n*1000:.0f} ms/point"
            )

        else:

            self._log(
                "Scan terminé sans résultat."
            )

        self.tabs.setCurrentIndex(
            0
        )

    # ==========================================================
    # PRPD
    # ==========================================================

    def _on_start_prpd(self) -> None:

        params = (
            self.config_panel.get_params()
        )

        gain_db = float(
            params["prpd_gain_db"]
        )

        freq_hz = float(
            params["prpd_freq_hz"]
        )

        freq_mhz = (
            freq_hz / 1e6
        )

        gain_cal = (
            self._get_cal_for_gain(
                gain_db
            )
        )

        if gain_cal is None:

            QMessageBox.critical(
                self,
                "Calibration manquante",
                "Calibration non disponible "
                "pour ce gain.",
            )

            return

        # ======================================================
        # IMPORTANT :
        # On efface les anciennes données PRPD / IQ
        # avant une nouvelle acquisition.
        # ======================================================

        self._last_prpd_phases = np.array(
            [],
            dtype=float,
        )

        self._last_prpd_amps = np.array(
            [],
            dtype=float,
        )

        self._last_prpd_info = {}

        self._last_prpd_spec_freqs = np.array(
            [],
            dtype=float,
        )

        self._last_prpd_spec_dbfs = np.array(
            [],
            dtype=float,
        )

        self._last_iq_samples = np.array(
            [],
            dtype=np.complex64,
        )

        self._last_prpd_params = dict(
            params
        )

        # ------------------------------------------------------

        self.prpd_panel.clear()

        self.progress_bar.setValue(
            0
        )

        self.progress_bar.setVisible(
            True
        )

        self.config_panel.set_prpd_running(
            True
        )

        self.status_lbl_main.setText(
            f"PRPD en cours — "
            f"{freq_mhz:.3f} MHz | "
            f"G={gain_db:.0f} dB | "
            f"USRP B200"
        )

        self._log(
            f"\n{'='*55}\n"
            f"Démarrage PRPD USRP\n"
            f"Freq : {freq_mhz:.3f} MHz | "
            f"Gain : {gain_db:.0f} dB | "
            f"Fs : {params['rate_hz']/1e6:.1f} MS/s | "
            f"Durée : {params['prpd_duration_s']:.3f} s × "
            f"{params['prpd_n_acq']} acq.\n"
            f"{'='*55}"
        )

        # ------------------------------------------------------

        self._prpd_thread = PrpdThread(
            params,
            gain_cal,
        )

        t = self._prpd_thread

        t.acq_done.connect(
            self._on_prpd_done
        )

        t.progress.connect(
            self.progress_bar.setValue
        )

        t.log_message.connect(
            self._log
        )

        t.error.connect(
            self._on_error
        )

        t.start()

    # ==========================================================

    def _on_stop_prpd(self) -> None:

        if (
            self._prpd_thread
            and self._prpd_thread.isRunning()
        ):

            self._prpd_thread.stop()

            self.status_lbl_main.setText(
                "PRPD interrompu."
            )

            self.config_panel.set_prpd_running(
                False
            )

    # ==========================================================
    # FIN ACQUISITION PRPD
    # ==========================================================

    def _on_prpd_done(
        self,
        phases: np.ndarray,
        amps: np.ndarray,
        info: dict,
        spec_freqs: np.ndarray,
        spec_dbfs: np.ndarray,
        iq_samples: np.ndarray,
    ) -> None:

        params = (
            self.config_panel.get_params()
        )

        freq_mhz = (
            params["prpd_freq_hz"]
            / 1e6
        )

        gain_db = (
            params["prpd_gain_db"]
        )

        # ======================================================
        # CONSERVATION PRPD
        # ======================================================

        self._last_prpd_phases = np.asarray(
            phases,
            dtype=float,
        ).copy()

        self._last_prpd_amps = np.asarray(
            amps,
            dtype=float,
        ).copy()

        self._last_prpd_info = dict(
            info
        )

        # ======================================================
        # CONSERVATION SPECTRE PRPD
        # ======================================================

        self._last_prpd_spec_freqs = np.asarray(
            spec_freqs,
            dtype=float,
        ).copy()

        self._last_prpd_spec_dbfs = np.asarray(
            spec_dbfs,
            dtype=float,
        ).copy()

        # ======================================================
        # CONSERVATION IQ BRUT
        #
        # Il s'agit exactement des IQ captés lors de
        # Démarrer PRPD.
        # ======================================================

        self._last_iq_samples = np.asarray(
            iq_samples,
            dtype=np.complex64,
        ).copy()

        self._last_prpd_params = dict(
            params
        )

        # ------------------------------------------------------

        n_iq = int(
            self._last_iq_samples.size
        )

        iq_size_mb = (
            self._last_iq_samples.nbytes
            / 1024**2
        )

        self._log(
            f"IQ PRPD conservé : "
            f"{n_iq:,} samples | "
            f"{iq_size_mb:.1f} MiB"
        )

        # ======================================================
        # AFFICHAGE PRPD
        # ======================================================

        self.prpd_panel.update_prpd(
            phases,
            amps,
            info,
            freq_mhz,
            gain_db,
        )

        self.progress_bar.setValue(
            100
        )

        self.config_panel.set_prpd_running(
            False
        )

        self.status_lbl_main.setText(
            f"PRPD terminé — "
            f"{info.get('n_pulses', 0)} pulses "
            f"| {freq_mhz:.3f} MHz "
            f"| {info.get('unit', '?')}"
        )

        self._log(
            f"\nPRPD terminé : "
            f"{info.get('n_pulses', 0)} pulses | "
            f"{n_iq:,} IQ samples | "
            f"{info.get('unit', '?')}"
        )

        self.tabs.setCurrentIndex(
            1
        )

    # ==========================================================
    # SÉLECTION FRÉQUENCE DEPUIS LE SPECTRE
    # ==========================================================

    def _on_freq_selected(
        self,
        freq_mhz: float,
    ) -> None:

        self.config_panel.set_prpd_freq(
            freq_mhz
        )

        self.status_lbl_main.setText(
            f"Fréquence PRPD sélectionnée : "
            f"{freq_mhz:.1f} MHz"
        )

        self._log(
            f"Fréquence PRPD → "
            f"{freq_mhz:.1f} MHz "
            f"(clic spectre)"
        )

        self._suggest_gain_for_freq(
            freq_mhz
        )

    # ==========================================================

    def _suggest_gain_for_freq(
        self,
        freq_mhz: float,
    ) -> None:

        """
        Met le gain du scan correspondant à cette fréquence
        dans la configuration PRPD lorsque possible.
        """

        if not self._scan_results:
            return

        freqs = np.asarray(
            [
                r.freq_mhz
                for r in self._scan_results
            ]
        )

        idx = int(
            np.argmin(
                np.abs(
                    freqs
                    - freq_mhz
                )
            )
        )

        res = self._scan_results[
            idx
        ]

        self._log(
            f"Gain mesuré pour "
            f"{freq_mhz:.1f} MHz : "
            f"{res.gain_db:.0f} dB "
            f"(max={res.max_dbm:.1f} dBm, "
            f"status={res.status})"
        )

        # ------------------------------------------------------
        # Le ConfigPanel possède déjà prpd_gain_combo
        # ------------------------------------------------------

        if not hasattr(
            self.config_panel,
            "prpd_gain_combo",
        ):
            return

        combo = (
            self.config_panel.prpd_gain_combo
        )

        # ------------------------------------------------------
        # Recherche du gain disponible le plus proche
        # directement à partir du combo.
        #
        # Cela évite une dépendance à core.auto_gain.
        # ------------------------------------------------------

        available = []

        for i in range(
            combo.count()
        ):

            value = combo.itemData(
                i
            )

            if value is not None:

                try:
                    available.append(
                        (
                            i,
                            float(value),
                        )
                    )

                except Exception:
                    pass

        if not available:
            return

        best_index, best_gain = min(
            available,
            key=lambda x:
                abs(
                    x[1]
                    - res.gain_db
                ),
        )

        combo.setCurrentIndex(
            best_index
        )

    # ==========================================================
    # EXPORT COMPLET
    # ==========================================================

    def _on_export(self) -> None:

        # ------------------------------------------------------
        # Vérification données disponibles
        # ------------------------------------------------------

        has_scan = (
            len(
                self._scan_results
            )
            > 0
        )

        has_prpd = (
            self._last_prpd_phases.size
            > 0
            and
            self._last_prpd_amps.size
            > 0
        )

        has_prpd_spectrum = (
            self._last_prpd_spec_freqs.size
            > 0
            and
            self._last_prpd_spec_dbfs.size
            > 0
        )

        has_iq = (
            self._last_iq_samples.size
            > 0
        )

        # ------------------------------------------------------

        if not any(
            [
                has_scan,
                has_prpd,
                has_prpd_spectrum,
                has_iq,
            ]
        ):

            QMessageBox.information(
                self,
                "Export",
                "Aucune donnée disponible "
                "à exporter.",
            )

            return

        # ======================================================
        # CHOIX DOSSIER
        # ======================================================

        parent_dir = (
            QFileDialog.getExistingDirectory(
                self,
                "Choisir le dossier d'export",
                str(
                    Path.home()
                ),
            )
        )

        if not parent_dir:
            return

        # ======================================================
        # DOSSIER MESURE
        # ======================================================

        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        export_dir = (
            Path(parent_dir)
            / f"measurement_{timestamp}"
        )

        try:

            export_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            files_saved = []

            # ==================================================
            # 1. SPECTRE
            # ==================================================

            if has_scan:

                spectrum_path = (
                    export_dir
                    / "spectrum.csv"
                )

                with open(
                    spectrum_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as f:

                    writer = csv.writer(
                        f
                    )

                    writer.writerow(
                        [
                            "freq_mhz",
                            "gain_db",
                            "max_dbfs",
                            "robust_dbfs",
                            "median_dbfs",
                            "max_dbm",
                            "robust_dbm",
                            "clipping_fraction",
                            "status",
                        ]
                    )

                    for r in self._scan_results:

                        writer.writerow(
                            [
                                f"{r.freq_mhz:.6f}",
                                f"{r.gain_db:.3f}",
                                f"{r.max_dbfs:.6f}",
                                f"{r.robust_dbfs:.6f}",
                                f"{r.median_dbfs:.6f}",
                                f"{r.max_dbm:.6f}",
                                f"{r.robust_dbm:.6f}",
                                f"{r.clipping_fraction:.8e}",
                                r.status,
                            ]
                        )

                files_saved.append(
                    "spectrum.csv"
                )

                self._log(
                    f"Spectre exporté → "
                    f"{spectrum_path}"
                )

            # ==================================================
            # 2. PRPD
            # ==================================================

            if has_prpd:

                prpd_path = (
                    export_dir
                    / "prpd.csv"
                )

                unit = (
                    self._last_prpd_info.get(
                        "unit",
                        "unknown",
                    )
                )

                with open(
                    prpd_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as f:

                    writer = csv.writer(
                        f
                    )

                    writer.writerow(
                        [
                            "phase_deg",
                            "amplitude",
                            "unit",
                        ]
                    )

                    for phase, amp in zip(
                        self._last_prpd_phases,
                        self._last_prpd_amps,
                    ):

                        writer.writerow(
                            [
                                f"{phase:.9f}",
                                f"{amp:.9f}",
                                unit,
                            ]
                        )

                files_saved.append(
                    "prpd.csv"
                )

                self._log(
                    f"PRPD exporté → "
                    f"{prpd_path}"
                )

            # ==================================================
            # 3. MATRICE PRPD 256 × 256
            # ==================================================

            if has_prpd:

                amp_min = float(
                    np.min(
                        self._last_prpd_amps
                    )
                )

                amp_max = float(
                    np.max(
                        self._last_prpd_amps
                    )
                )

                if amp_min == amp_max:
                    amp_max = (
                        amp_min + 1.0
                    )

                hist, phase_edges, amp_edges = (
                    np.histogram2d(
                        self._last_prpd_phases,
                        self._last_prpd_amps,
                        bins=(
                            256,
                            256,
                        ),
                        range=(
                            (
                                0.0,
                                360.0,
                            ),
                            (
                                amp_min,
                                amp_max,
                            ),
                        ),
                    )
                )

                matrix_path = (
                    export_dir
                    / "prpd_matrix_256x256.npz"
                )

                np.savez_compressed(
                    matrix_path,

                    counts=
                        hist.astype(
                            np.uint32
                        ),

                    phase_edges=
                        phase_edges,

                    amplitude_edges=
                        amp_edges,

                    unit=
                        str(
                            self._last_prpd_info.get(
                                "unit",
                                "unknown",
                            )
                        ),
                )

                files_saved.append(
                    "prpd_matrix_256x256.npz"
                )

                self._log(
                    f"Matrice PRPD exportée → "
                    f"{matrix_path}"
                )

            # ==================================================
            # 4. SPECTRE PRPD SI DISPONIBLE
            # ==================================================

            if has_prpd_spectrum:

                prpd_spec_path = (
                    export_dir
                    / "prpd_spectrum.csv"
                )

                with open(
                    prpd_spec_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as f:

                    writer = csv.writer(
                        f
                    )

                    writer.writerow(
                        [
                            "frequency_hz",
                            "frequency_mhz",
                            "amplitude_dbfs",
                        ]
                    )

                    for freq_hz, dbfs in zip(
                        self._last_prpd_spec_freqs,
                        self._last_prpd_spec_dbfs,
                    ):

                        writer.writerow(
                            [
                                f"{freq_hz:.6f}",
                                f"{freq_hz/1e6:.9f}",
                                f"{dbfs:.9f}",
                            ]
                        )

                files_saved.append(
                    "prpd_spectrum.csv"
                )

                self._log(
                    f"Spectre PRPD exporté → "
                    f"{prpd_spec_path}"
                )

            # ==================================================
            # 5. IQ BRUT PRPD
            # ==================================================

            if has_iq:

                iq_path = (
                    export_dir
                    / "iq_raw.npz"
                )

                params = (
                    self._last_prpd_params
                )

                # ------------------------------------------------
                # On sauvegarde le tableau complexe principal.
                #
                # I et Q ne sont PAS dupliqués dans le fichier,
                # car ils peuvent toujours être reconstruits :
                #
                # I = iq.real
                # Q = iq.imag
                #
                # Cela évite d'augmenter inutilement la taille.
                # ------------------------------------------------

                np.savez_compressed(
                    iq_path,

                    iq=
                        self._last_iq_samples,

                    sample_rate_hz=
                        float(
                            params.get(
                                "rate_hz",
                                np.nan,
                            )
                        ),

                    center_frequency_hz=
                        float(
                            params.get(
                                "prpd_freq_hz",
                                np.nan,
                            )
                        ),

                    gain_db=
                        float(
                            params.get(
                                "prpd_gain_db",
                                np.nan,
                            )
                        ),

                    duration_s=
                        float(
                            params.get(
                                "prpd_duration_s",
                                np.nan,
                            )
                        ),

                    n_acq=
                        int(
                            params.get(
                                "prpd_n_acq",
                                1,
                            )
                        ),
                )

                files_saved.append(
                    "iq_raw.npz"
                )

                self._log(
                    f"IQ brut PRPD exporté → "
                    f"{iq_path}"
                )

            # ==================================================
            # 6. METADATA
            # ==================================================

            params = (
                self._last_prpd_params
            )

            duration_s = params.get(
                "prpd_duration_s"
            )

            n_acq = params.get(
                "prpd_n_acq"
            )

            total_duration_s = None
            n_cycles = None

            if duration_s is not None:

                try:

                    duration_s = float(
                        duration_s
                    )

                    n_acq_int = int(
                        n_acq
                        if n_acq is not None
                        else 1
                    )

                    total_duration_s = (
                        duration_s
                        * n_acq_int
                    )

                    # 50 Hz -> période = 20 ms
                    n_cycles = (
                        total_duration_s
                        / 0.020
                    )

                except Exception:

                    total_duration_s = None
                    n_cycles = None

            # --------------------------------------------------

            metadata = {

                "application":
                    "GE IHM SDR",

                "version":
                    self.APP_VERSION,

                "date":
                    datetime.now().isoformat(),

                # ----------------------------------------------
                # Disponibilité
                # ----------------------------------------------

                "scan_available":
                    has_scan,

                "prpd_available":
                    has_prpd,

                "iq_available":
                    has_iq,

                # ----------------------------------------------
                # Scan
                # ----------------------------------------------

                "n_scan_points":
                    len(
                        self._scan_results
                    ),

                # ----------------------------------------------
                # PRPD
                # ----------------------------------------------

                "n_prpd_pulses":
                    int(
                        self._last_prpd_phases.size
                    ),

                "prpd_unit":
                    self._last_prpd_info.get(
                        "unit"
                    ),

                "prpd_frequency_hz":
                    self._safe_float(
                        params.get(
                            "prpd_freq_hz"
                        )
                    ),

                "prpd_gain_db":
                    self._safe_float(
                        params.get(
                            "prpd_gain_db"
                        )
                    ),

                "prpd_duration_s":
                    self._safe_float(
                        duration_s
                    ),

                "prpd_n_acq":
                    self._safe_int(
                        n_acq
                    ),

                "total_prpd_duration_s":
                    self._safe_float(
                        total_duration_s
                    ),

                "number_50hz_cycles":
                    self._safe_float(
                        n_cycles
                    ),

                # ----------------------------------------------
                # SDR
                # ----------------------------------------------

                "sample_rate_hz":
                    self._safe_float(
                        params.get(
                            "rate_hz"
                        )
                    ),

                "lowpass_cutoff_hz":
                    self._safe_float(
                        params.get(
                            "lowpass_cutoff_hz"
                        )
                    ),

                "channel":
                    params.get(
                        "channel"
                    ),

                "antenna":
                    params.get(
                        "antenna"
                    ),

                # ----------------------------------------------
                # IQ
                # ----------------------------------------------

                "n_iq_samples":
                    int(
                        self._last_iq_samples.size
                    ),

                "iq_dtype":
                    (
                        str(
                            self._last_iq_samples.dtype
                        )
                        if has_iq
                        else None
                    ),

                "iq_size_bytes":
                    (
                        int(
                            self._last_iq_samples.nbytes
                        )
                        if has_iq
                        else 0
                    ),

                "iq_size_mib":
                    (
                        float(
                            self._last_iq_samples.nbytes
                            / 1024**2
                        )
                        if has_iq
                        else 0.0
                    ),

                # ----------------------------------------------
                # Matrice PRPD
                # ----------------------------------------------

                "prpd_matrix_bins":
                    [
                        256,
                        256,
                    ],
            }

            # ==================================================
            # Sauvegarde JSON
            # ==================================================

            metadata_path = (
                export_dir
                / "metadata.json"
            )

            with open(
                metadata_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    metadata,
                    f,
                    indent=4,
                    ensure_ascii=False,
                    default=self._json_default,
                )

            files_saved.append(
                "metadata.json"
            )

            # ==================================================
            # FIN EXPORT
            # ==================================================

            self._log(
                f"\nExport complet terminé → "
                f"{export_dir}"
            )

            self.status_lbl_main.setText(
                f"Export : "
                f"{export_dir.name}"
            )

            message = (
                "Mesure sauvegardée dans :\n\n"
                f"{export_dir}\n\n"
                "Fichiers sauvegardés :\n"
            )

            message += "\n".join(
                f"• {name}"
                for name in files_saved
            )

            QMessageBox.information(
                self,
                "Export terminé",
                message,
            )

        except Exception as exc:

            self._log(
                f"[ERROR] Export : {exc}"
            )

            QMessageBox.critical(
                self,
                "Erreur export",
                str(exc),
            )

    # ==========================================================
    # HELPERS JSON
    # ==========================================================

    @staticmethod
    def _safe_float(
        value,
    ):

        if value is None:
            return None

        try:
            return float(
                value
            )

        except Exception:
            return None

    # ==========================================================

    @staticmethod
    def _safe_int(
        value,
    ):

        if value is None:
            return None

        try:
            return int(
                value
            )

        except Exception:
            return None

    # ==========================================================

    @staticmethod
    def _json_default(
        obj,
    ):

        if isinstance(
            obj,
            np.integer,
        ):

            return int(
                obj
            )

        if isinstance(
            obj,
            np.floating,
        ):

            return float(
                obj
            )

        if isinstance(
            obj,
            np.ndarray,
        ):

            return obj.tolist()

        return str(
            obj
        )

    # ==========================================================
    # LOG
    # ==========================================================

    def _log(
        self,
        msg: str,
    ) -> None:

        ts = time.strftime(
            "%H:%M:%S"
        )

        self.log_edit.append(
            "<span style='color:#484f58'>"
            f"[{ts}]"
            "</span> "
            f"{msg}"
        )

        sb = (
            self.log_edit.verticalScrollBar()
        )

        sb.setValue(
            sb.maximum()
        )

    # ==========================================================
    # ERREURS
    # ==========================================================

    def _on_error(
        self,
        msg: str,
    ) -> None:

        self._log(
            f"[ERROR] {msg}"
        )

        self.config_panel.set_scanning(
            False
        )

        self.config_panel.set_prpd_running(
            False
        )

        self.progress_bar.setVisible(
            False
        )

        self.status_lbl_main.setText(
            "Erreur — voir le journal"
        )

        QMessageBox.critical(
            self,
            "Erreur SDR",
            msg,
        )

    # ==========================================================
    # FERMETURE
    # ==========================================================

    def closeEvent(
        self,
        event,
    ) -> None:

        for thread in (
            self._scan_thread,
            self._prpd_thread,
        ):

            if (
                thread
                and thread.isRunning()
            ):

                thread.stop()

                thread.wait(
                    3000
                )

        event.accept()