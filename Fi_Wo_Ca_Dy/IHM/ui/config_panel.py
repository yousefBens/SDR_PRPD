#!/usr/bin/env python3
"""
ui/config_panel.py
------------------
Panneau de configuration latéral (QFrame).

Contient tous les paramètres USRP + boutons d'action.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.usrp_backend import DEFAULT_PARAMS
from core.auto_gain import GAINS_DB as GAINS_LIST


class ConfigPanel(QFrame):
    """
    Panneau latéral de configuration.

    Signaux :
      start_scan()      → lancer le scan
      stop_scan()       → arrêter
      start_prpd()      → lancer le PRPD
      stop_prpd()       → arrêter le PRPD
      export_results()  → exporter
      params_changed(dict)
    """

    start_scan     = pyqtSignal()
    stop_scan      = pyqtSignal()
    start_prpd     = pyqtSignal()
    stop_prpd      = pyqtSignal()
    export_results = pyqtSignal()
    params_changed = pyqtSignal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setObjectName("ConfigPanel")
        self.setFixedWidth(290)

        self._build_ui()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────
    # Style paramètres normalement fixes
    # ──────────────────────────────────────────────────────────

    def _set_fixed_param_style(self, widget) -> None:
        """
        Applique un style gris aux paramètres que l'utilisateur
        n'a normalement pas besoin de modifier.

        IMPORTANT :
        Le widget reste actif et totalement modifiable.
        """

        widget.setStyleSheet(
            """
            QLineEdit,
            QComboBox,
            QDoubleSpinBox,
            QSpinBox {
                background-color: #1b2027;
                color: #7d8590;

                border: 1px solid #30363d;
                border-radius: 5px;

                padding: 3px;
            }

            QLineEdit:hover,
            QComboBox:hover,
            QDoubleSpinBox:hover,
            QSpinBox:hover {
                background-color: #20262e;
                border: 1px solid #484f58;
            }

            QLineEdit:focus,
            QComboBox:focus,
            QDoubleSpinBox:focus,
            QSpinBox:focus {
                background-color: #222831;
                color: #c9d1d9;

                border: 1px solid #6e7681;
            }
            """
        )

    # ──────────────────────────────────────────────────────────
    # Construction UI
    # ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ------------------------------------------------------
        # Titre
        # ------------------------------------------------------

        title = QLabel("Configuration")

        title.setObjectName("PanelTitle")

        title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        outer.addWidget(title)

        # ------------------------------------------------------
        # Zone défilante
        # ------------------------------------------------------

        scroll = QScrollArea()

        scroll.setWidgetResizable(True)

        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        content = QWidget()

        content.setObjectName(
            "ConfigContent"
        )

        layout = QVBoxLayout(content)

        layout.setContentsMargins(
            10,
            10,
            10,
            10,
        )

        layout.setSpacing(8)

        # ------------------------------------------------------
        # Groupes
        # ------------------------------------------------------

        layout.addWidget(
            self._group_usrp()
        )

        layout.addWidget(
            self._group_scan()
        )

        layout.addWidget(
            self._group_agc()
        )

        layout.addWidget(
            self._group_prpd()
        )

        layout.addStretch()

        scroll.setWidget(content)

        outer.addWidget(scroll)

        # ------------------------------------------------------
        # Boutons
        # ------------------------------------------------------

        outer.addWidget(
            self._build_action_buttons()
        )

    # ──────────────────────────────────────────────────────────
    # Groupe USRP
    # ──────────────────────────────────────────────────────────

    def _group_usrp(self) -> QGroupBox:

        box = QGroupBox(
            "USRP B200"
        )

        box.setObjectName(
            "ConfigGroup"
        )

        lay = QVBoxLayout(box)

        # ======================================================
        # Numéro de série
        # ======================================================

        lay.addWidget(
            QLabel("Numéro de série :")
        )

        self.serial_edit = QLineEdit(
            DEFAULT_PARAMS["usrp_serial"]
        )

        self.serial_edit.setObjectName(
            "ConfigInput"
        )

        self.serial_edit.setPlaceholderText(
            "ex: 306BD15 (vide = auto)"
        )

        # Gris mais modifiable
        self._set_fixed_param_style(
            self.serial_edit
        )

        lay.addWidget(
            self.serial_edit
        )

        # ======================================================
        # Antenne
        # ======================================================

        lay.addWidget(
            QLabel("Antenne :")
        )

        self.antenna_combo = QComboBox()

        self.antenna_combo.addItems(
            [
                "RX2",
                "TX/RX",
            ]
        )

        # Gris mais modifiable
        self._set_fixed_param_style(
            self.antenna_combo
        )

        lay.addWidget(
            self.antenna_combo
        )

        # ======================================================
        # Débit I/Q
        # ======================================================

        lay.addWidget(
            QLabel("Débit I/Q (MSps) :")
        )

        self.rate_spin = QDoubleSpinBox()

        self.rate_spin.setRange(
            1.0,
            56.0,
        )

        self.rate_spin.setValue(
            DEFAULT_PARAMS["rate_hz"]
            / 1e6
        )

        self.rate_spin.setSingleStep(
            1.0
        )

        self.rate_spin.setSuffix(
            " MSps"
        )

        # Gris mais modifiable
        self._set_fixed_param_style(
            self.rate_spin
        )

        lay.addWidget(
            self.rate_spin
        )

        return box

    # ──────────────────────────────────────────────────────────
    # Groupe Scan
    # ──────────────────────────────────────────────────────────

    def _group_scan(self) -> QGroupBox:

        box = QGroupBox(
            "Scan spectral"
        )

        box.setObjectName(
            "ConfigGroup"
        )

        lay = QVBoxLayout(box)

        # ======================================================
        # F START
        # ======================================================

        lay.addWidget(
            QLabel("F_start (MHz) :")
        )

        self.fstart_spin = QDoubleSpinBox()

        self.fstart_spin.setRange(
            50.0,
            1900.0,
        )

        self.fstart_spin.setValue(
            DEFAULT_PARAMS["f_start_hz"]
            / 1e6
        )

        self.fstart_spin.setSuffix(
            " MHz"
        )

        # Gris mais modifiable
        self._set_fixed_param_style(
            self.fstart_spin
        )

        lay.addWidget(
            self.fstart_spin
        )

        # ======================================================
        # F STOP
        # ======================================================

        lay.addWidget(
            QLabel("F_stop (MHz) :")
        )

        self.fstop_spin = QDoubleSpinBox()

        self.fstop_spin.setRange(
            100.0,
            6000.0,
        )

        self.fstop_spin.setValue(
            DEFAULT_PARAMS["f_stop_hz"]
            / 1e6
        )

        self.fstop_spin.setSuffix(
            " MHz"
        )

        # Gris mais modifiable
        self._set_fixed_param_style(
            self.fstop_spin
        )

        lay.addWidget(
            self.fstop_spin
        )

        # ======================================================
        # PAS
        #
        # Paramètre laissé NORMAL
        # ======================================================

        lay.addWidget(
            QLabel("Pas (MHz) :")
        )

        self.step_spin = QDoubleSpinBox()

        self.step_spin.setRange(
            1.0,
            100.0,
        )

        self.step_spin.setValue(
            DEFAULT_PARAMS["step_hz"]
            / 1e6
        )

        self.step_spin.setSuffix(
            " MHz"
        )

        lay.addWidget(
            self.step_spin
        )

        # ======================================================
        # DURÉE UTILE
        # ======================================================

        lay.addWidget(
            QLabel("Durée utile (ms) :")
        )

        self.useful_spin = QDoubleSpinBox()

        self.useful_spin.setRange(
            1.0,
            500.0,
        )

        self.useful_spin.setValue(
            DEFAULT_PARAMS["useful_s"]
            * 1000
        )

        self.useful_spin.setSuffix(
            " ms"
        )

        # Gris mais modifiable
        self._set_fixed_param_style(
            self.useful_spin
        )

        lay.addWidget(
            self.useful_spin
        )

        # ======================================================
        # STABILISATION
        # ======================================================

        lay.addWidget(
            QLabel("Stabilisation (ms) :")
        )

        self.settle_spin = QDoubleSpinBox()

        self.settle_spin.setRange(
            20.0,
            500.0,
        )

        self.settle_spin.setValue(
            DEFAULT_PARAMS["settling_s"]
            * 1000
        )

        self.settle_spin.setSuffix(
            " ms"
        )

        # Gris mais modifiable
        self._set_fixed_param_style(
            self.settle_spin
        )

        lay.addWidget(
            self.settle_spin
        )

        return box

    # ──────────────────────────────────────────────────────────
    # Groupe AGC
    # ──────────────────────────────────────────────────────────

    def _group_agc(self) -> QGroupBox:

        box = QGroupBox(
            "Gain automatique (AGC)"
        )

        box.setObjectName(
            "ConfigGroup"
        )

        lay = QVBoxLayout(box)

        info = QLabel(
            "Gains disponibles : "
            "0 / 20 / 40 / 60 / 76 dB\n"

            "Départ : 40 dB\n"

            "Saturation si max > Pmax dBFS\n"

            "Saturation si min < Pmin dBFS\n"
        )

        info.setObjectName(
            "InfoLabel"
        )

        info.setWordWrap(True)

        lay.addWidget(info)

        return box

    # ──────────────────────────────────────────────────────────
    # Groupe PRPD
    # ──────────────────────────────────────────────────────────

    def _group_prpd(self) -> QGroupBox:

        box = QGroupBox(
            "PRPD"
        )

        box.setObjectName(
            "ConfigGroup"
        )

        lay = QVBoxLayout(box)

        # ======================================================
        # Fréquence
        # ======================================================

        lay.addWidget(
            QLabel(
                "Fréquence PRPD (MHz) :"
            )
        )

        self.prpd_freq_spin = (
            QDoubleSpinBox()
        )

        self.prpd_freq_spin.setRange(
            50.0,
            6000.0,
        )

        self.prpd_freq_spin.setDecimals(
            3
        )

        self.prpd_freq_spin.setValue(
            DEFAULT_PARAMS[
                "prpd_freq_hz"
            ]
            / 1e6
        )

        self.prpd_freq_spin.setSuffix(
            " MHz"
        )

        lay.addWidget(
            self.prpd_freq_spin
        )

        # ======================================================
        # Gain
        # ======================================================

        lay.addWidget(
            QLabel(
                "Gain PRPD (dB) :"
            )
        )

        self.prpd_gain_combo = QComboBox()

        for g in GAINS_LIST:

            self.prpd_gain_combo.addItem(
                f"{g:.0f} dB",
                g,
            )

        if (
            DEFAULT_PARAMS["prpd_gain_db"]
            in GAINS_LIST
        ):
            idx = GAINS_LIST.index(
                DEFAULT_PARAMS[
                    "prpd_gain_db"
                ]
            )
        else:
            idx = 2

        self.prpd_gain_combo.setCurrentIndex(
            idx
        )

        lay.addWidget(
            self.prpd_gain_combo
        )

        # ======================================================
        # Durée
        # ======================================================

        lay.addWidget(
            QLabel(
                "Durée acquisition (s) :"
            )
        )

        self.prpd_dur_spin = (
            QDoubleSpinBox()
        )

        self.prpd_dur_spin.setRange(
            0.5,
            60.0,
        )

        self.prpd_dur_spin.setValue(
            DEFAULT_PARAMS[
                "prpd_duration_s"
            ]
        )

        self.prpd_dur_spin.setSuffix(
            " s"
        )

        lay.addWidget(
            self.prpd_dur_spin
        )

        # ======================================================
        # Nombre acquisitions
        # ======================================================

        lay.addWidget(
            QLabel(
                "Nombre d'acquisitions :"
            )
        )

        self.prpd_nacq_spin = (
            QSpinBox()
        )

        self.prpd_nacq_spin.setRange(
            1,
            20,
        )

        self.prpd_nacq_spin.setValue(
            DEFAULT_PARAMS[
                "prpd_n_acq"
            ]
        )

        lay.addWidget(
            self.prpd_nacq_spin
        )

        # ======================================================
        # Offset fréquentiel
        # ======================================================

        lay.addWidget(
            QLabel(
                "Décalage fréquentiel "
                "(MHz) :"
            )
        )

        self.prpd_offset_spin = (
            QDoubleSpinBox()
        )

        self.prpd_offset_spin.setRange(
            -6.0,
            6.0,
        )

        self.prpd_offset_spin.setDecimals(
            2
        )

        self.prpd_offset_spin.setValue(
            DEFAULT_PARAMS[
                "prpd_f_offset_hz"
            ]
            / 1e6
        )

        self.prpd_offset_spin.setSuffix(
            " MHz"
        )

        lay.addWidget(
            self.prpd_offset_spin
        )

        return box

    # ──────────────────────────────────────────────────────────
    # Boutons
    # ──────────────────────────────────────────────────────────

    def _build_action_buttons(
        self
    ) -> QWidget:

        w = QWidget()

        w.setObjectName(
            "ActionBar"
        )

        lay = QVBoxLayout(w)

        lay.setContentsMargins(
            10,
            8,
            10,
            12,
        )

        lay.setSpacing(6)

        # ======================================================
        # SCAN
        # ======================================================

        row1 = QHBoxLayout()

        self.btn_scan = QPushButton(
            "Démarrer Scan"
        )

        self.btn_scan.setObjectName(
            "BtnPrimary"
        )

        self.btn_stop_scan = QPushButton(
            "Stop Scan"
        )

        self.btn_stop_scan.setObjectName(
            "BtnDanger"
        )

        self.btn_stop_scan.setEnabled(
            False
        )

        row1.addWidget(
            self.btn_scan
        )

        row1.addWidget(
            self.btn_stop_scan
        )

        lay.addLayout(row1)

        # ======================================================
        # PRPD
        # ======================================================

        row2 = QHBoxLayout()

        self.btn_prpd = QPushButton(
            "Démarrer PRPD"
        )

        self.btn_prpd.setObjectName(
            "BtnSecondary"
        )

        self.btn_stop_prpd = QPushButton(
            "Stop PRPD"
        )

        self.btn_stop_prpd.setObjectName(
            "BtnDanger"
        )

        self.btn_stop_prpd.setEnabled(
            False
        )

        row2.addWidget(
            self.btn_prpd
        )

        row2.addWidget(
            self.btn_stop_prpd
        )

        lay.addLayout(row2)

        # ======================================================
        # EXPORT
        # ======================================================

        self.btn_export = QPushButton(
            "Exporter résultats"
        )

        self.btn_export.setObjectName(
            "BtnExport"
        )

        lay.addWidget(
            self.btn_export
        )

        return w

    # ──────────────────────────────────────────────────────────
    # Connexions
    # ──────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:

        self.btn_scan.clicked.connect(
            self.start_scan
        )

        self.btn_stop_scan.clicked.connect(
            self.stop_scan
        )

        self.btn_prpd.clicked.connect(
            self.start_prpd
        )

        self.btn_stop_prpd.clicked.connect(
            self.stop_prpd
        )

        self.btn_export.clicked.connect(
            self.export_results
        )

    # ──────────────────────────────────────────────────────────
    # Lecture paramètres
    # ──────────────────────────────────────────────────────────

    def get_params(self) -> dict:

        return {

            "usrp_serial":
                self.serial_edit
                .text()
                .strip(),

            "antenna":
                self.antenna_combo
                .currentText(),

            "rate_hz":
                self.rate_spin
                .value()
                * 1e6,

            "f_start_hz":
                self.fstart_spin
                .value()
                * 1e6,

            "f_stop_hz":
                self.fstop_spin
                .value()
                * 1e6,

            "step_hz":
                self.step_spin
                .value()
                * 1e6,

            "useful_s":
                self.useful_spin
                .value()
                / 1000.0,

            "settling_s":
                self.settle_spin
                .value()
                / 1000.0,

            "prpd_freq_hz":
                self.prpd_freq_spin
                .value()
                * 1e6,

            "prpd_gain_db":
                float(
                    self.prpd_gain_combo
                    .currentData()
                ),

            "prpd_duration_s":
                self.prpd_dur_spin
                .value(),

            "prpd_n_acq":
                self.prpd_nacq_spin
                .value(),

            "prpd_f_offset_hz":
                self.prpd_offset_spin
                .value()
                * 1e6,
        }

    # ──────────────────────────────────────────────────────────
    # Fréquence PRPD
    # ──────────────────────────────────────────────────────────

    def set_prpd_freq(
        self,
        freq_mhz: float,
    ) -> None:
        """
        Appelé quand l'utilisateur
        clique sur le spectre.
        """

        self.prpd_freq_spin.setValue(
            freq_mhz
        )

    # ──────────────────────────────────────────────────────────
    # État scan
    # ──────────────────────────────────────────────────────────

    def set_scanning(
        self,
        scanning: bool,
    ) -> None:

        self.btn_scan.setEnabled(
            not scanning
        )

        self.btn_stop_scan.setEnabled(
            scanning
        )

        self.btn_prpd.setEnabled(
            not scanning
        )

    # ──────────────────────────────────────────────────────────
    # État PRPD
    # ──────────────────────────────────────────────────────────

    def set_prpd_running(
        self,
        running: bool,
    ) -> None:

        self.btn_prpd.setEnabled(
            not running
        )

        self.btn_stop_prpd.setEnabled(
            running
        )

        self.btn_scan.setEnabled(
            not running
        )