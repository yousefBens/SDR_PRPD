#!/usr/bin/env python3
"""
ui/main_window.py
-----------------
Fenêtre principale PyQt6 de l'IHM SDR.

Layout :
  - Gauche : ConfigPanel (paramètres + boutons)
  - Droite : QTabWidget
      · Onglet "Spectre"   → SpectrumPanel
      · Onglet "PRPD"      → PrpdPanel
      · Onglet "Journal"   → QTextEdit log

Barre de progression + status bar en bas.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon
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
    build_gain_calibration,
    load_power_calibration,
)
from core.usrp_backend import (
    DEFAULT_PARAMS,
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

    def __init__(self) -> None:
        super().__init__()
        self._scan_results: list[FreqResult] = []
        self._scan_thread = None
        self._prpd_thread = None
        self._cal_table:  dict | None = None
        self._all_cal:    dict = {}   # {gain_db: cal_dict}
        self._scan_start_time: float  = 0.0

        self.setWindowTitle(
            f"GE IHM — Détection Décharges Partielles SDR  v{self.APP_VERSION}"
        )
        self.setMinimumSize(1300, 780)
        self.resize(1520, 880)

        self._build_ui()
        self._connect_signals()
        self._load_calibration()

    # ──────────────────────────────────────────────────────────
    # Construction UI
    # ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        # Splitter horizontal principal
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Panneau de configuration
        self.config_panel = ConfigPanel()
        splitter.addWidget(self.config_panel)

        # Zone principale droite
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        # Onglets sans icônes textuelles/emojis
        self.tabs = QTabWidget()
        self.tabs.setObjectName("MainTabs")

        # Onglet Spectre
        self.spectrum_panel = SpectrumPanel()
        self.tabs.addTab(self.spectrum_panel, "Spectre dBm")

        # Onglet PRPD
        self.prpd_panel = PrpdPanel()
        self.tabs.addTab(self.prpd_panel, "PRPD")

        # Onglet Journal
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setObjectName("LogEdit")
        self.tabs.addTab(self.log_edit, "Journal")

        right_lay.addWidget(self.tabs)

        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(16)
        right_lay.addWidget(self.progress_bar)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Layout central
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.addWidget(splitter)

        # Barre de statut
        self._setup_status_bar()

    def _setup_status_bar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)

        self.status_lbl_main = QLabel("Prêt")
        self.status_lbl_freq = QLabel("")
        self.status_lbl_cal  = QLabel("Calibration : non chargée")

        sb.addWidget(self.status_lbl_main)
        sb.addPermanentWidget(self.status_lbl_freq)
        sb.addPermanentWidget(self.status_lbl_cal)

    # ──────────────────────────────────────────────────────────
    # Connexions signaux / slots
    # ──────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.config_panel.start_scan.connect(self._on_start_scan)
        self.config_panel.stop_scan.connect(self._on_stop_scan)
        self.config_panel.start_prpd.connect(self._on_start_prpd)
        self.config_panel.stop_prpd.connect(self._on_stop_prpd)
        self.config_panel.export_results.connect(self._on_export)
        self.spectrum_panel.freq_selected.connect(self._on_freq_selected)

    # ──────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────

    def _load_calibration(self) -> None:
        try:
            self._cal_table = load_power_calibration()
            self._all_cal   = build_all_gain_calibrations(self._cal_table)
            gains_str = ", ".join(f"{g:.0f}" for g in sorted(self._all_cal.keys()))
            self.status_lbl_cal.setText(f"Calibration : [{gains_str}] dB")
            self._log(
                f"Calibration chargée — gains disponibles : {gains_str} dB"
            )
        except FileNotFoundError as exc:
            self.status_lbl_cal.setText("Calibration : fichier introuvable")
            self._log(f"[WARN] Calibration non disponible : {exc}")
            QMessageBox.warning(
                self,
                "Calibration manquante",
                f"Fichier de calibration non trouvé :\n{exc}\n\n"
                "Vérifiez l'emplacement du fichier pickle de calibration.",
            )
        except Exception as exc:
            self.status_lbl_cal.setText("Calibration : erreur")
            self._log(f"[ERROR] Calibration : {exc}")

    def _get_cal_for_gain(self, gain_db: float) -> dict | None:
        if not self._all_cal:
            return None
        if gain_db in self._all_cal:
            return self._all_cal[gain_db]
        # Gain le plus proche
        arr = np.asarray(sorted(self._all_cal.keys()))
        nearest = float(arr[np.argmin(np.abs(arr - gain_db))])
        return self._all_cal.get(nearest)

    # ──────────────────────────────────────────────────────────
    # Scan spectral
    # ──────────────────────────────────────────────────────────

    def _on_start_scan(self) -> None:
        params = self.config_panel.get_params()

        if not self._all_cal:
            QMessageBox.critical(
                self,
                "Calibration manquante",
                "Impossible de lancer le scan sans calibration.\n"
                "Vérifiez le fichier pickle de calibration.",
            )
            return

        self._scan_results.clear()
        self.spectrum_panel.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.config_panel.set_scanning(True)
        self._scan_start_time = time.perf_counter()

        n_freqs = int(
            (params["f_stop_hz"] - params["f_start_hz"]) / params["step_hz"]
        ) + 1
        self.status_lbl_main.setText(
            f"Scan en cours — {n_freqs} fréquences | USRP B200"
        )
        self._log(
            f"\n{'='*55}\n"
            f"Démarrage scan USRP\n"
            f"  F : {params['f_start_hz']/1e6:.0f} → "
            f"{params['f_stop_hz']/1e6:.0f} MHz | "
            f"Pas {params['step_hz']/1e6:.0f} MHz | "
            f"{n_freqs} points\n"
            f"{'='*55}"
        )

        self._scan_thread = ScanThread(params, self._all_cal)

        t = self._scan_thread
        t.result_ready.connect(self._on_scan_result)
        t.progress.connect(self.progress_bar.setValue)
        t.scan_done.connect(self._on_scan_done)
        t.log_message.connect(self._log)
        t.error.connect(self._on_error)
        t.start()

    def _on_stop_scan(self) -> None:
        if self._scan_thread and self._scan_thread.isRunning():
            self._scan_thread.stop()
            self.status_lbl_main.setText("Scan interrompu.")

    def _on_scan_result(self, result: FreqResult) -> None:
        self._scan_results.append(result)
        self.spectrum_panel.add_result(result)
        self.status_lbl_freq.setText(
            f"{result.freq_mhz:.1f} MHz | "
            f"G={result.gain_db:.0f} dB | "
            f"{result.max_dbm:.1f} dBm"
        )

    def _on_scan_done(self, results: list[FreqResult]) -> None:
        self._scan_results = results
        elapsed = time.perf_counter() - self._scan_start_time
        n = len(results)
        self.progress_bar.setValue(100)
        self.config_panel.set_scanning(False)
        self.status_lbl_main.setText(
            f"Scan terminé — {n} points en {elapsed:.1f} s"
        )
        self._log(
            f"\nScan terminé : {n} points | {elapsed:.1f} s | "
            f"{elapsed/n*1000:.0f} ms/point"
        )
        self.tabs.setCurrentIndex(0)

    # ──────────────────────────────────────────────────────────
    # PRPD
    # ──────────────────────────────────────────────────────────

    def _on_start_prpd(self) -> None:
        params     = self.config_panel.get_params()
        gain_db    = params["prpd_gain_db"]
        freq_hz    = params["prpd_freq_hz"]
        freq_mhz   = freq_hz / 1e6

        gain_cal = self._get_cal_for_gain(gain_db)
        if gain_cal is None:
            QMessageBox.critical(
                self,
                "Calibration manquante",
                "Calibration non disponible pour ce gain.",
            )
            return

        self.prpd_panel.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.config_panel.set_prpd_running(True)

        self.status_lbl_main.setText(
            f"PRPD en cours — {freq_mhz:.3f} MHz | "
            f"G={gain_db:.0f} dB | USRP B200"
        )
        self._log(
            f"\n{'='*55}\n"
            f"Démarrage PRPD USRP\n"
            f"  Freq : {freq_mhz:.3f} MHz | Gain : {gain_db:.0f} dB | "
            f"Durée : {params['prpd_duration_s']:.1f} s × "
            f"{params['prpd_n_acq']} acq.\n"
            f"{'='*55}"
        )

        self._prpd_thread = PrpdThread(params, gain_cal)

        t = self._prpd_thread
        t.acq_done.connect(self._on_prpd_done)
        t.progress.connect(self.progress_bar.setValue)
        t.log_message.connect(self._log)
        t.error.connect(self._on_error)
        t.start()

    def _on_stop_prpd(self) -> None:
        if self._prpd_thread and self._prpd_thread.isRunning():
            self._prpd_thread.stop()
            self.status_lbl_main.setText("PRPD interrompu.")
            self.config_panel.set_prpd_running(False)

    def _on_prpd_done(
        self,
        phases: np.ndarray,
        amps:   np.ndarray,
        info:   dict,
        spec_freqs: np.ndarray,
        spec_dbfs:  np.ndarray,
    ) -> None:
        params   = self.config_panel.get_params()
        freq_mhz = params["prpd_freq_hz"] / 1e6
        gain_db  = params["prpd_gain_db"]

        self.prpd_panel.update_prpd(phases, amps, info, freq_mhz, gain_db)

        self.progress_bar.setValue(100)
        self.config_panel.set_prpd_running(False)
        self.status_lbl_main.setText(
            f"PRPD terminé — {info.get('n_pulses', 0)} pulses "
            f"| {freq_mhz:.3f} MHz | {info.get('unit', '?')}"
        )
        self._log(
            f"\nPRPD terminé : {info.get('n_pulses', 0)} pulses "
            f"| unité {info.get('unit', '?')}"
        )
        self.tabs.setCurrentIndex(1)

    # ──────────────────────────────────────────────────────────
    # Sélection fréquence PRPD via clic spectre
    # ──────────────────────────────────────────────────────────

    def _on_freq_selected(self, freq_mhz: float) -> None:
        self.config_panel.set_prpd_freq(freq_mhz)
        self.status_lbl_main.setText(
            f"Fréquence PRPD sélectionnée : {freq_mhz:.1f} MHz"
        )
        self._log(f"Fréquence PRPD → {freq_mhz:.1f} MHz (clic spectre)")

        self._suggest_gain_for_freq(freq_mhz)

    def _suggest_gain_for_freq(self, freq_mhz: float) -> None:
        """Suggère le gain optimal mesuré pendant le scan."""
        if not self._scan_results:
            return

        freqs = np.asarray([r.freq_mhz for r in self._scan_results])
        idx   = int(np.argmin(np.abs(freqs - freq_mhz)))
        res   = self._scan_results[idx]

        self._log(
            f"  Gain recommandé pour {freq_mhz:.1f} MHz : "
            f"{res.gain_db:.0f} dB "
            f"(max={res.max_dbm:.1f} dBm, status={res.status})"
        )
        combo = self.config_panel.prpd_gain_combo
        from core.auto_gain import GAINS_DB as GAINS_LIST
        nearest_gain = min(GAINS_LIST, key=lambda g: abs(g - res.gain_db))
        for i in range(combo.count()):
            if combo.itemData(i) == nearest_gain:
                combo.setCurrentIndex(i)
                break

    # ──────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────

    def _on_export(self) -> None:
        if not self._scan_results:
            QMessageBox.information(
                self, "Export", "Aucun résultat de scan à exporter."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Exporter le spectre",
            str(Path.home() / "spectrum_export.csv"),
            "CSV (*.csv);;Tous (*)",
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "freq_mhz", "gain_db",
                    "max_dbfs", "robust_dbfs", "median_dbfs",
                    "max_dbm", "robust_dbm",
                    "clipping_fraction", "status",
                ])
                for r in self._scan_results:
                    writer.writerow([
                        f"{r.freq_mhz:.4f}",
                        f"{r.gain_db:.1f}",
                        f"{r.max_dbfs:.3f}",
                        f"{r.robust_dbfs:.3f}",
                        f"{r.median_dbfs:.3f}",
                        f"{r.max_dbm:.3f}",
                        f"{r.robust_dbm:.3f}",
                        f"{r.clipping_fraction:.2e}",
                        r.status,
                    ])
            self._log(f"Export CSV → {path}")
            self.status_lbl_main.setText(f"Export : {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Erreur export", str(exc))

    # ──────────────────────────────────────────────────────────
    # Log
    # ──────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_edit.append(f"<span style='color:#484f58'>[{ts}]</span> {msg}")
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ──────────────────────────────────────────────────────────
    # Erreurs
    # ──────────────────────────────────────────────────────────

    def _on_error(self, msg: str) -> None:
        self._log(f"[ERROR] {msg}")
        self.config_panel.set_scanning(False)
        self.config_panel.set_prpd_running(False)
        self.progress_bar.setVisible(False)
        self.status_lbl_main.setText("Erreur — voir le journal")
        QMessageBox.critical(self, "Erreur SDR", msg)

    # ──────────────────────────────────────────────────────────
    # Fermeture propre
    # ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        for thread in (self._scan_thread, self._prpd_thread):
            if thread and thread.isRunning():
                thread.stop()
                thread.wait(3000)
        event.accept()
