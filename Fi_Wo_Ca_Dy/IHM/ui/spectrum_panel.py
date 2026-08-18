#!/usr/bin/env python3
"""
ui/spectrum_panel.py
--------------------
Widget matplotlib pour l'affichage du spectre calibré.

Axes :
  Gauche  : Puissance estimée à l'entrée RX2 (dBm)
  Droite  : Gain RX sélectionné (dB) — courbe en escalier

Interactivité :
  - Barre d'outils Matplotlib (Zoom rectangle, Pan, Reset, Enregistrer)
  - Zoom & Dézoom à la molette de la souris
  - Double-clic pour réinitialiser le zoom
  - Suivi dynamique des coordonnées (Fréquence MHz, Puissance dBm) au survol
  - Clic gauche sur le spectre → sélectionne la fréquence PRPD
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from core.usrp_backend import FreqResult

# Couleurs
C_MAX    = "#00d4ff"   # cyan vif → MAX dBm
C_ROBUST = "#7b61ff"   # violet → percentile 99.99%
C_GAIN   = "#ff8c42"   # orange → gain
C_SAT    = "#ff4444"   # rouge  → saturation
C_WEAK   = "#ffcc00"   # jaune  → signal faible
BG       = "#0d1117"
AX_BG    = "#161b22"
GRID     = "#21262d"
TEXT     = "#e6edf3"


class SpectrumPanel(QWidget):
    """Widget du spectre dBm + courbe gain avec zoom interactif et coordonnées."""

    freq_selected = pyqtSignal(float)   # MHz

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._freqs: list[float] = []
        self._max_dbm: list[float] = []
        self._robust_dbm: list[float] = []
        self._gains: list[float] = []
        self._statuses: list[str] = []
        self._selected_freq_mhz: float | None = None
        self._user_zoomed: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Barre supérieure d'outils + coordonnées
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(5, 2, 5, 2)

        self.fig = Figure(figsize=(12, 5), facecolor=BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("""
            QToolBar { background-color: #161b22; border: none; padding: 2px; }
            QToolButton { color: #e6edf3; background-color: #21262d; border-radius: 4px; padding: 3px; margin: 1px; }
            QToolButton:hover { background-color: #30363d; }
        """)
        top_layout.addWidget(self.toolbar)

        hint_lbl = QLabel("Molette: Zoom in/out | Double-clic: Réinitialiser")
        hint_lbl.setStyleSheet("color: #8b949e; font-size: 11px; padding-left: 10px;")
        top_layout.addWidget(hint_lbl)

        top_layout.addStretch()

        self.coord_lbl = QLabel("Freq: — | Pwr: —")
        self.coord_lbl.setStyleSheet("color: #00d4ff; font-weight: bold; padding-right: 10px; font-size: 11px;")
        top_layout.addWidget(self.coord_lbl)

        layout.addWidget(top_bar)
        layout.addWidget(self.canvas)

        # Deux axes
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()

        self._style_axes()
        self.fig.tight_layout(pad=2.0)

        # Événements Matplotlib (clic, survol, molette)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _style_axes(self) -> None:
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor(AX_BG)
            ax.tick_params(colors=TEXT, labelsize=9)
            ax.spines[:].set_color(GRID)

        self.ax1.set_xlabel("Fréquence centrale RX (MHz)", color=TEXT, fontsize=10)
        self.ax1.set_ylabel("Puissance estimée RX2 (dBm)", color=C_MAX, fontsize=10)
        self.ax2.set_ylabel("Gain RX sélectionné (dB)", color=C_GAIN, fontsize=10)
        self.ax1.tick_params(axis="y", colors=C_MAX)
        self.ax2.tick_params(axis="y", colors=C_GAIN)
        self.ax1.set_title(
            "Spectre calibré — MAX temporel | Gain auto RX",
            color=TEXT, fontsize=11, fontweight="bold", pad=8,
        )
        self.ax1.grid(color=GRID, alpha=0.6, linewidth=0.5)

    def clear(self) -> None:
        """Réinitialise les données et le graphe."""
        self._freqs.clear()
        self._max_dbm.clear()
        self._robust_dbm.clear()
        self._gains.clear()
        self._statuses.clear()
        self._user_zoomed = False
        self.coord_lbl.setText("Freq: — | Pwr: —")
        self.ax1.cla()
        self.ax2.cla()
        self._style_axes()
        self.fig.canvas.draw_idle()

    def add_result(self, result: FreqResult) -> None:
        """Ajoute un point de fréquence (appelé depuis le thread)."""
        self._freqs.append(result.freq_mhz)
        self._max_dbm.append(result.max_dbm)
        self._robust_dbm.append(result.robust_dbm)
        self._gains.append(result.gain_db)
        self._statuses.append(result.status)
        self._redraw()

    def set_selected_freq(self, freq_mhz: float) -> None:
        self._selected_freq_mhz = freq_mhz
        self._redraw()

    def _redraw(self) -> None:
        if not self._freqs:
            return

        cur_xlim1 = self.ax1.get_xlim() if self._user_zoomed else None
        cur_ylim1 = self.ax1.get_ylim() if self._user_zoomed else None

        f  = np.asarray(self._freqs)
        m  = np.asarray(self._max_dbm)
        r  = np.asarray(self._robust_dbm)
        g  = np.asarray(self._gains)
        st = self._statuses

        self.ax1.cla()
        self.ax2.cla()
        self._style_axes()

        # ── Zones colorées saturation / faible SNR ─────────────
        sat_mask  = np.array([s in ("SAT@MIN_GAIN",) for s in st])
        weak_mask = np.array([s in ("WEAK@MAX_GAIN",) for s in st])

        for mask, col, alpha in [
            (sat_mask,  C_SAT,  0.15),
            (weak_mask, C_WEAK, 0.12),
        ]:
            if mask.any():
                in_zone = False
                z_start = None
                for i, v in enumerate(mask):
                    if v and not in_zone:
                        in_zone = True
                        z_start = f[i]
                    elif not v and in_zone:
                        in_zone = False
                        self.ax1.axvspan(
                            z_start, f[i - 1], alpha=alpha, color=col,
                        )
                if in_zone:
                    self.ax1.axvspan(z_start, f[-1], alpha=alpha, color=col)

        # ── Spectre dBm ────────────────────────────────────────
        valid = np.isfinite(m)
        if valid.any():
            self.ax1.plot(
                f[valid], m[valid],
                color=C_MAX, linewidth=1.8, label="MAX temporel calibré",
                zorder=4,
            )

        valid_r = np.isfinite(r)
        if valid_r.any():
            self.ax1.plot(
                f[valid_r], r[valid_r],
                color=C_ROBUST, linewidth=1.3, linestyle="--",
                label="Percentile 99.99% calibré", zorder=3,
            )

        # ── Gain en escalier ───────────────────────────────────
        if len(g) > 0:
            self.ax2.step(
                f, g,
                color=C_GAIN, linewidth=1.5, where="post",
                label="Gain sélectionné", zorder=2, alpha=0.85,
            )
            self.ax2.fill_between(
                f, 0, g,
                step="post", color=C_GAIN, alpha=0.06,
            )
            self.ax2.set_yticks(sorted(set(g.tolist())))
            self.ax2.set_ylim(
                max(0.0, float(g.min()) - 15),
                float(g.max()) + 10,
            )

        # ── Fréquence PRPD sélectionnée ───────────────────────
        if self._selected_freq_mhz is not None and len(f) > 0:
            self.ax1.axvline(
                self._selected_freq_mhz,
                color="#ff4081", linewidth=1.8, linestyle=":",
                label=f"PRPD : {self._selected_freq_mhz:.1f} MHz", zorder=5,
            )

        # ── Légende ────────────────────────────────────────────
        legend_patches = []
        if sat_mask.any():
            legend_patches.append(
                Patch(color=C_SAT, alpha=0.5, label="Zone saturation")
            )
        if weak_mask.any():
            legend_patches.append(
                Patch(color=C_WEAK, alpha=0.5, label="Zone faible SNR")
            )

        lines1, labels1 = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax1.legend(
            lines1 + lines2 + legend_patches,
            labels1 + labels2 + [p.get_label() for p in legend_patches],
            loc="upper left",
            fontsize=8,
            facecolor=BG,
            labelcolor=TEXT,
            framealpha=0.85,
        )

        if self._user_zoomed and cur_xlim1 is not None and cur_ylim1 is not None:
            self.ax1.set_xlim(cur_xlim1)
            self.ax1.set_ylim(cur_ylim1)
        elif len(f) > 1:
            self.ax1.set_xlim(f.min(), f.max())

        self.fig.tight_layout(pad=2.0)
        self.fig.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes not in (self.ax1, self.ax2):
            return

        # Si double-clic : réinitialiser le zoom
        if event.dblclick:
            self._user_zoomed = False
            self._redraw()
            return

        # Ignore la sélection de fréquence si un outil de la toolbar (zoom / pan) est actif
        if getattr(self.toolbar, "mode", "") != "":
            return

        if event.xdata is None:
            return

        freq_mhz = float(event.xdata)
        self._selected_freq_mhz = freq_mhz
        self.freq_selected.emit(freq_mhz)
        self._redraw()

    def _on_mouse_move(self, event) -> None:
        if event.inaxes in (self.ax1, self.ax2) and event.xdata is not None and event.ydata is not None:
            freq = event.xdata
            pwr  = event.ydata
            self.coord_lbl.setText(
                f"Freq: {freq:.2f} MHz | Val: {pwr:.2f} dBm"
            )

    def _on_scroll(self, event) -> None:
        """Zoom in / Zoom out à la molette de la souris."""
        if event.inaxes not in (self.ax1, self.ax2) or event.xdata is None or event.ydata is None:
            return

        base_scale = 1.25
        if event.button == "up":
            scale_factor = 1 / base_scale   # Zoom In
        elif event.button == "down":
            scale_factor = base_scale       # Zoom Out
        else:
            return

        self._user_zoomed = True
        ax = self.ax1
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        new_w = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        rel_x = (event.xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (event.ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([event.xdata - new_w * rel_x, event.xdata + new_w * (1 - rel_x)])
        ax.set_ylim([event.ydata - new_h * rel_y, event.ydata + new_h * (1 - rel_y)])
        self.canvas.draw_idle()
