"""Acquisition IQ temporelle avec UHD et sauvegarde au format NPZ."""
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import uhd

# Paramètres à adapter
DEVICE_ARGS = "serial=306BD15"  # "" pour sélection automatique
ANTENNE = "RX2"
CANAL = 0
FREQUENCE_HZ = 300e6
SAMPLING_RATE_HZ = 12e6
GAIN_DB = 20
DUREE_S = 0.02
DECALAGE_LO_HZ = 2e6
TEMPS_STABILISATION_S = 0.15
DOSSIER_SORTIE = Path("/home/yousef/Documents/testing_scripts/GEVernova/Fi_Wo_Ca_Dy/save_raw/raw_time_data")


def creer_recepteur(device_args, canal, antenne, sampling_rate_hz):
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_rx_rate(sampling_rate_hz, canal)
    usrp.set_rx_antenna(antenne, canal)
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [canal]
    return usrp, usrp.get_rx_stream(stream_args)


def acquerir_et_enregistrer(usrp, streamer, canal, frequence_hz, gain_db,
                             duree_s, decalage_lo_hz, stabilisation_s, fichier):
    if not np.isfinite(duree_s) or duree_s <= 0:
        raise ValueError("La durée d'acquisition doit être positive et finie")
    usrp.set_rx_gain(gain_db, canal)
    usrp.set_rx_freq(uhd.types.TuneRequest(frequence_hz, decalage_lo_hz), canal)
    time.sleep(stabilisation_s)
    if "lo_locked" in usrp.get_rx_sensor_names(canal):
        if not usrp.get_rx_sensor("lo_locked", canal).to_bool():
            raise RuntimeError("LO SDR non verrouillé")

    debit = float(usrp.get_rx_rate(canal))
    nombre = max(1, round(duree_s * debit))
    iq = np.empty(nombre, dtype=np.complex64)
    buffer = np.empty((1, streamer.get_max_num_samps()), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    commande = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    commande.num_samps = nombre
    commande.stream_now = True
    streamer.issue_stream_cmd(commande)
    total = 0
    date_utc = datetime.now(timezone.utc).isoformat()
    try:
        while total < nombre:
            recus = streamer.recv(buffer, metadata, 3.0)
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                raise RuntimeError(f"Erreur UHD : {metadata.strerror()}")
            if recus <= 0:
                raise RuntimeError("Aucun échantillon reçu")
            n = min(recus, nombre - total)
            iq[total:total + n] = buffer[0, :n]
            total += n
    finally:
        streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

    fichier = Path(fichier)
    fichier.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fichier, iq_samples=iq, requested_center_frequency_hz=frequence_hz,
             center_frequency_hz=float(usrp.get_rx_freq(canal)), sampling_rate_hz=debit,
             gain_db=float(usrp.get_rx_gain(canal)),
             bandwidth_hz=float(usrp.get_rx_bandwidth(canal)),
             antenna=usrp.get_rx_antenna(canal), channel=canal,
             lo_offset_hz=decalage_lo_hz, timestamp_utc=date_utc)
    print(f"Enregistré : {fichier} ({len(iq)} échantillons)")


def nom_valeur(valeur):
    """Format compact pour les valeurs dans les noms de fichiers."""
    return f"{valeur:g}"


def main():
    usrp, streamer = creer_recepteur(DEVICE_ARGS, CANAL, ANTENNE, SAMPLING_RATE_HZ)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nom = (f"acquisition_fc{nom_valeur(FREQUENCE_HZ)}Hz_"
           f"fs{nom_valeur(SAMPLING_RATE_HZ)}Hz_gain{nom_valeur(GAIN_DB)}dB_"
           f"duree{nom_valeur(DUREE_S)}s_lo{nom_valeur(DECALAGE_LO_HZ)}Hz_"
           f"ch{CANAL}_{horodatage}.npz")
    acquerir_et_enregistrer(usrp, streamer, CANAL, FREQUENCE_HZ, GAIN_DB,
                             DUREE_S, DECALAGE_LO_HZ, TEMPS_STABILISATION_S,
                             DOSSIER_SORTIE / nom)


if __name__ == "__main__":
    main()
