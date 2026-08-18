#!/usr/bin/env python3
"""
ui/prpd_panel.py
----------------
Widget matplotlib pour l'affichage PRPD (Phase-Resolved Partial Discharge).

Affiche :
  - Scatter plot plein écran : phase 0→360° vs amplitude (dBm)
  - Courbe de référence 50 Hz

Interactivité :
  - Barre d'outils Matplotlib (Zoom rectangle, Pan, Reset, Enregistrer)
  - Zoom & Dézoom à la molette de la souris
  - Double-clic pour réinitialiser le zoom
  - Affichage dynamique des coordonnées (Phase °, Amplitude dBm) au survol de la souris
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

BG    = "#0d1117"
AX_BG = "#161b22"
GRID  = "#21262d"
TEXT  = "#e6edf3"

C_SCATTER = "#00d4ff"
C_REF     = "#ff8c42"


class PrpdPanel(QWidget):
    """Widget PRPD — scatter plot plein écran avec zoom interactif et coordonnées."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._phases: np.ndarray  = np.array([])
        self._amps:   np.ndarray  = np.array([])
        self._info:   dict        = {}
        self._freq_mhz: float     = 0.0
        self._gain_db:  float     = 0.0
        self._cbar_scatter        = None
        self._user_zoomed: bool   = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Barre supérieure d'outils + coordonnées
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(5, 2, 5, 2)

        self.fig = Figure(figsize=(14, 5), facecolor=BG)
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

        # Indication raccourcis
        hint_lbl = QLabel("Molette: Zoom in/out | Double-clic: Réinitialiser")
        hint_lbl.setStyleSheet("color: #8b949e; font-size: 11px; padding-left: 10px;")
        top_layout.addWidget(hint_lbl)

        top_layout.addStretch()

        self.coord_lbl = QLabel("Phase: — | Amp: —")
        self.coord_lbl.setStyleSheet("color: #00d4ff; font-weight: bold; padding-right: 10px; font-size: 11px;")
        top_layout.addWidget(self.coord_lbl)

        layout.addWidget(top_bar)
        layout.addWidget(self.canvas)

        self.ax_scatter = self.fig.add_subplot(1, 1, 1)
        self._style_axes()
        self.fig.tight_layout(pad=2.0)

        # Connexions des événements Matplotlib (souris, molette, clic)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def _style_axes(self) -> None:
        self.ax_scatter.set_facecolor(AX_BG)
        self.ax_scatter.tick_params(colors=TEXT, labelsize=9)
        self.ax_scatter.spines[:].set_color(GRID)

    def _clear_colorbars(self) -> None:
        if getattr(self, "_cbar_scatter", None) is not None:
            try:
                self._cbar_scatter.remove()
            except Exception:
                pass
            self._cbar_scatter = None

        # Nettoyage de tout axe résiduel créé par fig.colorbar
        for ax in list(self.fig.axes):
            if ax != self.ax_scatter:
                try:
                    ax.remove()
                except Exception:
                    pass

    def clear(self) -> None:
        self._phases = np.array([])
        self._amps   = np.array([])
        self._info   = {}
        self._user_zoomed = False
        self._clear_colorbars()
        self.ax_scatter.cla()
        self._style_axes()
        self.coord_lbl.setText("Phase: — | Amp: —")
        self.fig.canvas.draw_idle()

    def update_prpd(
        self,
        phases: np.ndarray,
        amps: np.ndarray,
        info: dict,
        freq_mhz: float,
        gain_db: float,
    ) -> None:
        self._phases   = np.asarray(phases)
        self._amps     = np.asarray(amps)
        self._info     = info
        self._freq_mhz = freq_mhz
        self._gain_db  = gain_db
        self._redraw()

    def _redraw(self) -> None:
        # Conserver la vue zoomée si l'utilisateur a personnalisé le zoom
        cur_xlim = self.ax_scatter.get_xlim() if self._user_zoomed else None
        cur_ylim = self.ax_scatter.get_ylim() if self._user_zoomed else None

        self._clear_colorbars()
        self.ax_scatter.cla()
        self._style_axes()

        unit      = self._info.get("unit", "dBm")
        n_pulses  = self._info.get("n_pulses", 0)
        n_acq     = self._info.get("n_acq", 1)

        title_line = (
            f"Phase-Resolved Partial Discharge (PRPD) — {self._freq_mhz:.3f} MHz | "
            f"Gain {self._gain_db:.0f} dB | "
            f"Cal: {unit} | "
            f"{n_pulses} pulses / {n_acq} acq."
        )

        # ── Scatter ───────────────────────────────────────────
        if len(self._phases) > 0:
            sc = self.ax_scatter.scatter(
                self._phases,
                self._amps,
                s=10,
                c=self._amps,
                cmap="plasma",
                alpha=0.7,
                zorder=3,
            )
            self._cbar_scatter = self.fig.colorbar(
                sc, ax=self.ax_scatter, label=f"Amplitude ({unit})"
            )

            # Référence 50 Hz
            y_min = float(self._amps.min())
            y_max = float(self._amps.max())
            y_span = (y_max - y_min) if y_max > y_min else 20.0
            ref_phase = np.linspace(0, 360, 500)
            ref_sig   = np.sin(np.radians(ref_phase)) * (0.3 * y_span)
            ref_sig  += y_min + 0.5 * y_span

            self.ax_scatter.plot(
                ref_phase, ref_sig,
                color=C_REF, linewidth=1.5, label="Réf. 50 Hz", zorder=4,
            )
            self.ax_scatter.legend(
                fontsize=8, facecolor=BG, labelcolor=TEXT, loc="upper right"
            )

        self.ax_scatter.set_title(title_line, color=TEXT, fontsize=10, fontweight="bold", pad=8)
        self.ax_scatter.set_xlabel("Phase réseau (°)", color=TEXT, fontsize=9)
        self.ax_scatter.set_ylabel(f"Amplitude ({unit})", color=TEXT, fontsize=9)

        if self._user_zoomed and cur_xlim is not None and cur_ylim is not None:
            self.ax_scatter.set_xlim(cur_xlim)
            self.ax_scatter.set_ylim(cur_ylim)
        else:
            self.ax_scatter.set_xlim(0, 360)
            self.ax_scatter.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])

        self.ax_scatter.grid(color=GRID, alpha=0.5, linewidth=0.5)

        self.fig.tight_layout(pad=2.0)
        self.fig.canvas.draw_idle()

    def _on_mouse_move(self, event) -> None:
        if event.inaxes == self.ax_scatter and event.xdata is not None and event.ydata is not None:
            unit = self._info.get("unit", "dBm")
            self.coord_lbl.setText(
                f"Phase: {event.xdata:.1f}° | Amp: {event.ydata:.2f} {unit}"
            )

    def _on_scroll(self, event) -> None:
        """Zoom in / Zoom out avec la molette de la souris."""
        if event.inaxes != self.ax_scatter or event.xdata is None or event.ydata is None:
            return

        base_scale = 1.25
        if event.button == "up":
            scale_factor = 1 / base_scale   # Zoom In
        elif event.button == "down":
            scale_factor = base_scale       # Zoom Out
        else:
            return

        self._user_zoomed = True
        ax = self.ax_scatter
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        new_w = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        rel_x = (event.xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (event.ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([event.xdata - new_w * rel_x, event.xdata + new_w * (1 - rel_x)])
        ax.set_ylim([event.ydata - new_h * rel_y, event.ydata + new_h * (1 - rel_y)])
        self.canvas.draw_idle()

    def _on_click(self, event) -> None:
        """Réinitialise le zoom au double-clic."""
        if event.inaxes == self.ax_scatter and event.dblclick:
            self._user_zoomed = False
            self._redraw()
