#!/usr/bin/env python3
"""
ihm_main.py
-----------
Point d'entrée de l'IHM SDR — Détection de Décharges Partielles.

Usage :
    python ihm_main.py

Dépendances :
    pip install PyQt6 matplotlib numpy scipy
    uhd (installé avec le pilote USRP)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Résolution des imports relatifs depuis le dossier GE_IHM
sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


def load_stylesheet(app: QApplication) -> None:
    qss_path = Path(__file__).parent / "assets" / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
    else:
        print(f"[WARN] Feuille de style non trouvée : {qss_path}")


def setup_palette(app: QApplication) -> None:
    """Palette sombre Qt de base (complément du QSS)."""
    palette = QPalette()
    bg  = QColor("#0d1117")
    fg  = QColor("#e6edf3")
    alt = QColor("#161b22")
    hl  = QColor("#1f6feb")

    for role, color in [
        (QPalette.ColorRole.Window,          bg),
        (QPalette.ColorRole.WindowText,      fg),
        (QPalette.ColorRole.Base,            alt),
        (QPalette.ColorRole.AlternateBase,   bg),
        (QPalette.ColorRole.Text,            fg),
        (QPalette.ColorRole.Button,          alt),
        (QPalette.ColorRole.ButtonText,      fg),
        (QPalette.ColorRole.Highlight,       hl),
        (QPalette.ColorRole.HighlightedText, fg),
        (QPalette.ColorRole.ToolTipBase,     alt),
        (QPalette.ColorRole.ToolTipText,     fg),
    ]:
        palette.setColor(role, color)

    app.setPalette(palette)


def main() -> None:
    # Attributs haute résolution pour écrans HiDPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("GE IHM SDR")
    app.setApplicationDisplayName("GE IHM — Décharges Partielles")
    app.setOrganizationName("GE Vernova")

    # Police de base
    font = QFont("Inter", 12)
    if not font.exactMatch():
        font = QFont("Segoe UI", 12)
    app.setFont(font)

    # Style
    setup_palette(app)
    load_stylesheet(app)

    # Fenêtre principale
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
