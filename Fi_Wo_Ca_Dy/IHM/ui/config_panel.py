#!/usr/bin/env python3
"""
ui/config_panel.py
------------------
Panneau de configuration de l'IHM.
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


# Gains utilisés/calibrés
GAINS_LIST = [0.0, 20.0, 40.0, 60.0, 76.0]


class ConfigPanel(QFrame):

    start_scan = pyqtSignal()
    stop_scan = pyqtSignal()

    start_prpd = pyqtSignal()
    stop_prpd = pyqtSignal()

    export_results = pyqtSignal()

    params_changed = pyqtSignal(dict)


    def __init__(self, parent=None):

        super().__init__(parent)

        self.setObjectName("ConfigPanel")
        self.setFixedWidth(290)

        self._build_ui()
        self._connect_signals()


    # =========================================================
    # STYLE PARAMÈTRES FIXES / RAREMENT MODIFIÉS
    # =========================================================

    def _set_fixed_param_style(self, widget):

        widget.setStyleSheet(
            """
            background-color: #e8e8e8;
            color: #555555;
            """
        )


    # =========================================================
    # CONSTRUCTION UI
    # =========================================================

    def _build_ui(self):

        outer = QVBoxLayout(self)

        outer.setContentsMargins(
            0,
            0,
            0,
            0
        )

        outer.setSpacing(0)


        # -----------------------------------------------------
        # Titre
        # -----------------------------------------------------

        title = QLabel(
            "Configuration"
        )

        title.setObjectName(
            "PanelTitle"
        )

        title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        outer.addWidget(
            title
        )


        # -----------------------------------------------------
        # Scroll
        # -----------------------------------------------------

        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )

        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )


        content = QWidget()

        content.setObjectName(
            "ConfigContent"
        )


        layout = QVBoxLayout(
            content
        )

        layout.setContentsMargins(
            10,
            10,
            10,
            10
        )

        layout.setSpacing(8)


        # -----------------------------------------------------
        # Groupes
        # -----------------------------------------------------

        layout.addWidget(
            self._group_usrp()
        )

        layout.addWidget(
            self._group_scan()
        )

        layout.addWidget(
            self._group_prpd()
        )

        layout.addStretch()


        scroll.setWidget(
            content
        )

        outer.addWidget(
            scroll
        )


        # -----------------------------------------------------
        # Boutons
        # -----------------------------------------------------

        outer.addWidget(
            self._build_action_buttons()
        )


    # =========================================================
    # USRP
    # =========================================================

    def _group_usrp(self):

        box = QGroupBox(
            "USRP B200"
        )

        box.setObjectName(
            "ConfigGroup"
        )

        lay = QVBoxLayout(
            box
        )


        # -----------------------------------------------------
        # Serial
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Numéro de série :"
            )
        )

        self.serial_edit = QLineEdit(
            DEFAULT_PARAMS[
                "usrp_serial"
            ]
        )

        self.serial_edit.setPlaceholderText(
            "ex: 306BD15"
        )

        self._set_fixed_param_style(
            self.serial_edit
        )

        lay.addWidget(
            self.serial_edit
        )


        # -----------------------------------------------------
        # Antenne
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Antenne :"
            )
        )

        self.antenna_combo = QComboBox()

        self.antenna_combo.addItems(
            [
                "RX2",
                "TX/RX"
            ]
        )

        self.antenna_combo.setCurrentText(
            DEFAULT_PARAMS[
                "antenna"
            ]
        )

        self._set_fixed_param_style(
            self.antenna_combo
        )

        lay.addWidget(
            self.antenna_combo
        )


        # -----------------------------------------------------
        # Sample rate
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Débit I/Q :"
            )
        )

        self.rate_spin = QDoubleSpinBox()

        self.rate_spin.setRange(
            1.0,
            56.0
        )

        self.rate_spin.setDecimals(
            2
        )

        self.rate_spin.setValue(
            DEFAULT_PARAMS[
                "rate_hz"
            ] / 1e6
        )

        self.rate_spin.setSuffix(
            " MSps"
        )

        self._set_fixed_param_style(
            self.rate_spin
        )

        lay.addWidget(
            self.rate_spin
        )


        return box


    # =========================================================
    # SCAN SPECTRAL
    # =========================================================

    def _group_scan(self):

        box = QGroupBox(
            "Scan spectral"
        )

        box.setObjectName(
            "ConfigGroup"
        )

        lay = QVBoxLayout(
            box
        )


        # -----------------------------------------------------
        # F START
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "F_start :"
            )
        )

        self.fstart_spin = QDoubleSpinBox()

        self.fstart_spin.setRange(
            50.0,
            6000.0
        )

        self.fstart_spin.setValue(
            DEFAULT_PARAMS[
                "f_start_hz"
            ] / 1e6
        )

        self.fstart_spin.setSuffix(
            " MHz"
        )

        self._set_fixed_param_style(
            self.fstart_spin
        )

        lay.addWidget(
            self.fstart_spin
        )


        # -----------------------------------------------------
        # F STOP
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "F_stop :"
            )
        )

        self.fstop_spin = QDoubleSpinBox()

        self.fstop_spin.setRange(
            50.0,
            6000.0
        )

        self.fstop_spin.setValue(
            DEFAULT_PARAMS[
                "f_stop_hz"
            ] / 1e6
        )

        self.fstop_spin.setSuffix(
            " MHz"
        )

        self._set_fixed_param_style(
            self.fstop_spin
        )

        lay.addWidget(
            self.fstop_spin
        )


        # -----------------------------------------------------
        # PAS
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Pas :"
            )
        )

        self.step_spin = QDoubleSpinBox()

        self.step_spin.setRange(
            1.0,
            100.0
        )

        self.step_spin.setValue(
            DEFAULT_PARAMS[
                "step_hz"
            ] / 1e6
        )

        self.step_spin.setSuffix(
            " MHz"
        )

        lay.addWidget(
            self.step_spin
        )


        # -----------------------------------------------------
        # GAIN FIXE DU SCAN
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Gain spectre :"
            )
        )

        self.scan_gain_combo = QComboBox()

        for gain in GAINS_LIST:

            self.scan_gain_combo.addItem(
                f"{gain:.0f} dB",
                gain
            )

        default_gain = float(
            DEFAULT_PARAMS.get(
                "scan_gain_db",
                40.0
            )
        )

        for i in range(
            self.scan_gain_combo.count()
        ):

            gain = float(
                self.scan_gain_combo.itemData(i)
            )

            if gain == default_gain:

                self.scan_gain_combo.setCurrentIndex(
                    i
                )

                break

        lay.addWidget(
            self.scan_gain_combo
        )


        # -----------------------------------------------------
        # LOW PASS CUTOFF
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Low-pass cutoff :"
            )
        )

        self.cutoff_spin = QDoubleSpinBox()

        self.cutoff_spin.setRange(
            0.1,
            28.0
        )

        self.cutoff_spin.setDecimals(
            2
        )

        self.cutoff_spin.setSingleStep(
            0.1
        )

        self.cutoff_spin.setValue(
            DEFAULT_PARAMS[
                "lowpass_cutoff_hz"
            ] / 1e6
        )

        self.cutoff_spin.setSuffix(
            " MHz"
        )

        lay.addWidget(
            self.cutoff_spin
        )


        # -----------------------------------------------------
        # DURÉE UTILE
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Durée utile :"
            )
        )

        self.useful_spin = QDoubleSpinBox()

        self.useful_spin.setRange(
            1.0,
            500.0
        )

        self.useful_spin.setValue(
            DEFAULT_PARAMS[
                "useful_s"
            ] * 1000
        )

        self.useful_spin.setSuffix(
            " ms"
        )

        self._set_fixed_param_style(
            self.useful_spin
        )

        lay.addWidget(
            self.useful_spin
        )


        # -----------------------------------------------------
        # STABILISATION
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Stabilisation :"
            )
        )

        self.settle_spin = QDoubleSpinBox()

        self.settle_spin.setRange(
            1.0,
            500.0
        )

        self.settle_spin.setValue(
            DEFAULT_PARAMS[
                "settling_s"
            ] * 1000
        )

        self.settle_spin.setSuffix(
            " ms"
        )

        self._set_fixed_param_style(
            self.settle_spin
        )

        lay.addWidget(
            self.settle_spin
        )


        return box


    # =========================================================
    # PRPD
    # =========================================================

    def _group_prpd(self):

        box = QGroupBox(
            "PRPD"
        )

        box.setObjectName(
            "ConfigGroup"
        )

        lay = QVBoxLayout(
            box
        )


        # -----------------------------------------------------
        # Fréquence PRPD
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Fréquence PRPD :"
            )
        )

        self.prpd_freq_spin = QDoubleSpinBox()

        self.prpd_freq_spin.setRange(
            50.0,
            6000.0
        )

        self.prpd_freq_spin.setDecimals(
            3
        )

        self.prpd_freq_spin.setValue(
            DEFAULT_PARAMS[
                "prpd_freq_hz"
            ] / 1e6
        )

        self.prpd_freq_spin.setSuffix(
            " MHz"
        )

        lay.addWidget(
            self.prpd_freq_spin
        )


        # -----------------------------------------------------
        # Gain PRPD
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Gain PRPD :"
            )
        )

        self.prpd_gain_combo = QComboBox()

        for gain in GAINS_LIST:

            self.prpd_gain_combo.addItem(
                f"{gain:.0f} dB",
                gain
            )


        default_gain = float(
            DEFAULT_PARAMS[
                "prpd_gain_db"
            ]
        )

        for i in range(
            self.prpd_gain_combo.count()
        ):

            gain = float(
                self.prpd_gain_combo.itemData(i)
            )

            if gain == default_gain:

                self.prpd_gain_combo.setCurrentIndex(
                    i
                )

                break


        lay.addWidget(
            self.prpd_gain_combo
        )


        # -----------------------------------------------------
        # Durée PRPD
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Durée acquisition :"
            )
        )

        self.prpd_dur_spin = QDoubleSpinBox()

        self.prpd_dur_spin.setRange(
            0.5,
            60.0
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


        # -----------------------------------------------------
        # Nombre acquisitions
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Nombre d'acquisitions :"
            )
        )

        self.prpd_nacq_spin = QSpinBox()

        self.prpd_nacq_spin.setRange(
            1,
            20
        )

        self.prpd_nacq_spin.setValue(
            DEFAULT_PARAMS[
                "prpd_n_acq"
            ]
        )

        lay.addWidget(
            self.prpd_nacq_spin
        )


        # -----------------------------------------------------
        # Offset
        # -----------------------------------------------------

        lay.addWidget(
            QLabel(
                "Décalage fréquentiel :"
            )
        )

        self.prpd_offset_spin = QDoubleSpinBox()

        self.prpd_offset_spin.setRange(
            -6.0,
            6.0
        )

        self.prpd_offset_spin.setDecimals(
            2
        )

        self.prpd_offset_spin.setValue(
            DEFAULT_PARAMS[
                "prpd_f_offset_hz"
            ] / 1e6
        )

        self.prpd_offset_spin.setSuffix(
            " MHz"
        )

        lay.addWidget(
            self.prpd_offset_spin
        )


        return box


    # =========================================================
    # BOUTONS
    # =========================================================

    def _build_action_buttons(self):

        widget = QWidget()

        widget.setObjectName(
            "ActionBar"
        )

        lay = QVBoxLayout(
            widget
        )

        lay.setContentsMargins(
            10,
            8,
            10,
            12
        )

        lay.setSpacing(6)


        # -----------------------------------------------------
        # Scan
        # -----------------------------------------------------

        row_scan = QHBoxLayout()

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


        row_scan.addWidget(
            self.btn_scan
        )

        row_scan.addWidget(
            self.btn_stop_scan
        )

        lay.addLayout(
            row_scan
        )


        # -----------------------------------------------------
        # PRPD
        # -----------------------------------------------------

        row_prpd = QHBoxLayout()

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


        row_prpd.addWidget(
            self.btn_prpd
        )

        row_prpd.addWidget(
            self.btn_stop_prpd
        )

        lay.addLayout(
            row_prpd
        )


        # -----------------------------------------------------
        # Export
        # -----------------------------------------------------

        self.btn_export = QPushButton(
            "Exporter résultats"
        )

        self.btn_export.setObjectName(
            "BtnExport"
        )

        lay.addWidget(
            self.btn_export
        )


        return widget


    # =========================================================
    # CONNEXIONS
    # =========================================================

    def _connect_signals(self):

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


    # =========================================================
    # RÉCUPÉRATION DES PARAMÈTRES
    # =========================================================

    def get_params(self):

        return {

            # USRP
            "usrp_serial":
                self.serial_edit.text().strip(),

            "antenna":
                self.antenna_combo.currentText(),

            "rate_hz":
                self.rate_spin.value()
                * 1e6,


            # Scan
            "f_start_hz":
                self.fstart_spin.value()
                * 1e6,

            "f_stop_hz":
                self.fstop_spin.value()
                * 1e6,

            "step_hz":
                self.step_spin.value()
                * 1e6,

            "scan_gain_db":
                float(
                    self.scan_gain_combo.currentData()
                ),

            "lowpass_cutoff_hz":
                self.cutoff_spin.value()
                * 1e6,

            "useful_s":
                self.useful_spin.value()
                / 1000.0,

            "settling_s":
                self.settle_spin.value()
                / 1000.0,


            # PRPD
            "prpd_freq_hz":
                self.prpd_freq_spin.value()
                * 1e6,

            "prpd_gain_db":
                float(
                    self.prpd_gain_combo.currentData()
                ),

            "prpd_duration_s":
                self.prpd_dur_spin.value(),

            "prpd_n_acq":
                self.prpd_nacq_spin.value(),

            "prpd_f_offset_hz":
                self.prpd_offset_spin.value()
                * 1e6,
        }


    # =========================================================
    # FRÉQUENCE PRPD CHOISIE DEPUIS LE SPECTRE
    # =========================================================

    def set_prpd_freq(
        self,
        freq_mhz
    ):

        self.prpd_freq_spin.setValue(
            freq_mhz
        )


    # =========================================================
    # ÉTAT SCAN
    # =========================================================

    def set_scanning(
        self,
        scanning
    ):

        self.btn_scan.setEnabled(
            not scanning
        )

        self.btn_stop_scan.setEnabled(
            scanning
        )

        self.btn_prpd.setEnabled(
            not scanning
        )


    # =========================================================
    # ÉTAT PRPD
    # =========================================================

    def set_prpd_running(
        self,
        running
    ):

        self.btn_prpd.setEnabled(
            not running
        )

        self.btn_stop_prpd.setEnabled(
            running
        )

        self.btn_scan.setEnabled(
            not running
        )