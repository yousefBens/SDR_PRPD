"""Balayage UHD : une acquisition IQ et un fichier NPZ par fréquence."""
from datetime import datetime
from pathlib import Path
import math

import uhd  # Nécessite les bindings Python UHD.
from acquisition_temporelle import creer_recepteur, acquerir_et_enregistrer, nom_valeur

# Configuration de la SDR
DEVICE_ARGS = "serial=306BD15"  # "" pour sélection automatique
ANTENNE = "RX2"
CANAL = 0
GAIN_DB = 20
SAMPLING_RATE_HZ = 12e6       # Échantillons par seconde
DECALAGE_LO_HZ = 2e6
TEMPS_STABILISATION_S = 0.15

# Balayage en Hz ; notation décimale comme 100e6 acceptée.
# La fin est incluse si elle tombe sur un pas.
FREQUENCE_DEBUT_HZ = 100e6
FREQUENCE_FIN_HZ = 600e6
PAS_HZ = SAMPLING_RATE_HZ
DUREE_ACQUISITION_S = 0.02
DOSSIER_SORTIE = Path("/home/yousef/Documents/testing_scripts/GEVernova/Fi_Wo_Ca_Dy/save_raw/Balayage_data")


def main():
    if (not all(math.isfinite(x) for x in (FREQUENCE_DEBUT_HZ, FREQUENCE_FIN_HZ, PAS_HZ))
            or PAS_HZ <= 0 or FREQUENCE_DEBUT_HZ <= 0 or FREQUENCE_FIN_HZ < FREQUENCE_DEBUT_HZ):
        raise ValueError("Vérifier la plage de fréquences et le pas")
    nombre_points = math.floor((FREQUENCE_FIN_HZ - FREQUENCE_DEBUT_HZ) / PAS_HZ + 1e-12) + 1
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nom = (f"balayage_debut{nom_valeur(FREQUENCE_DEBUT_HZ)}Hz_"
           f"fin{nom_valeur(FREQUENCE_FIN_HZ)}Hz_pas{nom_valeur(PAS_HZ)}Hz_"
           f"fs{nom_valeur(SAMPLING_RATE_HZ)}Hz_gain{nom_valeur(GAIN_DB)}dB_"
           f"duree{nom_valeur(DUREE_ACQUISITION_S)}s_lo{nom_valeur(DECALAGE_LO_HZ)}Hz_"
           f"ch{CANAL}_{horodatage}")
    dossier = DOSSIER_SORTIE / nom
    dossier.mkdir(parents=True, exist_ok=False)
    usrp, streamer = creer_recepteur(DEVICE_ARGS, CANAL, ANTENNE, SAMPLING_RATE_HZ)
    for index in range(nombre_points):
        frequence = FREQUENCE_DEBUT_HZ + index * PAS_HZ
        fichier = dossier / f"{index:04d}_fc{nom_valeur(frequence)}Hz.npz"
        acquerir_et_enregistrer(usrp, streamer, CANAL, frequence, GAIN_DB,
                                 DUREE_ACQUISITION_S, DECALAGE_LO_HZ,
                                 TEMPS_STABILISATION_S, fichier)
    print(f"Balayage terminé : {dossier}")


if __name__ == "__main__":
    main()
