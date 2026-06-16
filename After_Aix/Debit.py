import os
import time
import numpy as np
import uhd

# ==============================================================================
# CONFIGURATION MATÉRIELLE DIRECTE
# ==============================================================================
FREQ = 1965e6    # Fréquence centrale SDR (1.965 GHz)
RATE = 12e6      # Taux d'échantillonnage (12 MSps)
DURATION = 3.0   # Durée de capture en secondes
GAIN = 76        # Amplification RF (dB)
CHANNEL = 0      # Premier canal de l'USRP
ANTENNA = "RX2"  # Port physique de réception

OUTPUT_DIR = "./Main_figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1) SYNCHRONISATION SUR LE 50 HZ VIA EXT-PPS
# ==============================================================================
def sync_on_external_50hz_pps(usrp):
    """
    Verrouille la base de temps de l'USRP sur le signal 50 Hz externe.
    Force une remise à zéro (0.0s) au prochain front montant.
    """
    usrp.set_clock_source("internal")
    usrp.set_time_source("external")

    # Attente du passage d'un front PPS (ton signal 50 Hz secteur)
    time_last = usrp.get_time_last_pps().get_real_secs()
    while usrp.get_time_last_pps().get_real_secs() == time_last:
        time.sleep(0.001)

    # Programmation de la mise à zéro de l'horloge au tout prochain front
    usrp.set_time_next_pps(uhd.types.TimeSpec(0.0))
    time.sleep(0.05)  # Petite pause pour laisser le front passer et appliquer le reset

# ==============================================================================
# 2) CAPTURE BRUTE & MESURE DES PERFORMANCES NUMÉRIQUES
# ==============================================================================
def rx_and_measure_throughput(usrp, freq, rate, duration, gain, antenna):
    """
    Déclenche la capture brute et calcule le débit réel du flux de données.
    """
    num_samps = int(duration * rate)

    # Application des paramètres radio au composant d'entrée (Frontend)
    usrp.set_rx_rate(rate, CHANNEL)
    usrp.set_rx_freq(uhd.types.TuneRequest(freq), CHANNEL)
    usrp.set_rx_gain(gain, CHANNEL)
    usrp.set_rx_antenna(antenna, CHANNEL)
    usrp.set_rx_dc_offset(True, CHANNEL)
    usrp.set_rx_iq_balance(True, CHANNEL)
    time.sleep(0.5)

    # sc16 sur le câble (4 octets/sample) -> fc32 dans Python (8 octets/sample)
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [CHANNEL]
    rx_streamer = usrp.get_rx_stream(stream_args)
    rx_md = uhd.types.RXMetadata()

    # Allocation de la mémoire RAM pour stocker les échantillons IQ
    received = np.zeros(num_samps, dtype=np.complex64)

    # Calcul du rendez-vous futur calé pile sur un début de cycle 50Hz (multiple de 20ms)
    current_time = usrp.get_time_now().get_real_secs()
    future_time = current_time + 0.1
    start_time = np.ceil(future_time / 0.02) * 0.02

    print(f"Temps SDR actuel = {current_time:.6f} s")
    print(f"Enregistrement planifié pour t = {start_time:.6f} s (Début de cycle)")

    # Envoi de la commande de streaming planifiée au FPGA
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samps
    stream_cmd.stream_now = False
    stream_cmd.time_spec = uhd.types.TimeSpec(start_time)
    rx_streamer.issue_stream_cmd(stream_cmd)

    # Tampon de réception intermédiaire (Buffer de transfert)
    buff = np.zeros((1, 4096), dtype=np.complex64)
    total = 0
    t_start_real = None

    # Initialisation des variables de benchmark
    bytes_per_sample_cable = 4  # Format sc16 (2 octets pour I, 2 octets pour Q)
    t_start_perf = time.time()
    dernier_affichage = time.time()
    echantillons_depuis_affichage = 0

    print("\n=== DÉBUT DE L'ANALYSE DE DÉBIT RÉEL ===")

    while total < num_samps:
        # Récupération des paquets d'échantillons depuis le buffer réseau/USB
        n = rx_streamer.recv(buff, rx_md, timeout=5.0)

        # Vérification des erreurs de transmission du bus
        if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
            print(f" ❌ Alerte UHD : {rx_md.strerror()}")
            if rx_md.error_code == uhd.types.RXMetadataErrorCode.overflow:
                print("⚠️ OVERFLOW ! Le PC est trop lent, des données sont perdues.")
            continue

        if n > 0:
            if t_start_real is None:
                t_start_real = rx_md.time_spec.get_real_secs()

            # Remplissage du tableau final dans la RAM
            end = min(total + n, num_samps)
            received[total:end] = buff[0, :end - total]
            total = end

            echantillons_depuis_affichage += n
            t_actuel = time.time()

            # Calcul et affichage dynamique du débit toutes les 0.5 secondes
            if (t_actuel - dernier_affichage) >= 0.5:
                dt = t_actuel - dernier_affichage
                debit_sps = echantillons_depuis_affichage / dt
                debit_mbytes_s = (debit_sps * bytes_per_sample_cable) / 1e6
                debit_mbits_s = debit_mbytes_s * 8

                print(f"-> [Progression: {total/num_samps*100:.1f}%] "
                      f"Débit mesuré : {debit_mbytes_s:.2f} Mo/s ({debit_mbits_s:.1f} Mbit/s) | "
                      f"Cadence : {debit_sps/1e6:.2f} MSps")

                dernier_affichage = t_actuel
                echantillons_depuis_affichage = 0

    t_total_calculé = time.time() - t_start_perf

    # ==============================================================================
    # 3) RAPPORT DE PERFORMANCE FINAL
    # ==============================================================================
    print("\n=== RAPPORT DE PERFORMANCES GLOBAL ===")
    print(f"Durée théorique de capture attendue : {duration} s")
    print(f"Durée réelle d'exécution mesurée   : {t_total_calculé:.4f} s")

    total_octets_cable = total * bytes_per_sample_cable
    debit_moyen_mo_s = (total_octets_cable / t_total_calculé) / 1e6
    debit_moyen_mbit_s = debit_moyen_mo_s * 8

    print(f"Total de données transférées sur le câble : {total_octets_cable/1e6:.2f} Mo")
    print(f"Débit MOYEN global constaté               : {debit_moyen_mo_s:.2f} Mo/s ({debit_moyen_mbit_s:.1f} Mbit/s)")

    if abs(duration - t_total_calculé) < 0.1:
        print("✅ Statut du lien : Flux 100% stable. Aucune perte.")
    else:
        print("❌ Statut du lien : Des latences ou des ralentissements système ont été détectés.")

    return received[:total], t_start_real

# ==============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ==============================================================================
def main():
    print("Connexion au périphérique USRP...")
    try:
        usrp = uhd.usrp.MultiUSRP()
    except Exception as e:
        print(f"❌ Connexion impossible. Vérifiez le câble ou l'adresse IP. Erreur: {e}")
        return

    # Étape 1 : Alignement temporel sur l'horloge externe (50 Hz)
    print("\n[Étape 1] Synchronisation temporelle sur le secteur...")
    sync_on_external_50hz_pps(usrp)

    # Étape 2 : Lancement du streaming et calcul du débit en continu
    print("\n[Étape 2] Lancement de la capture brute IQ...")
    rx_signal, t_start = rx_and_measure_throughput(
        usrp=usrp,
        freq=FREQ,
        rate=RATE,
        duration=DURATION,
        gain=GAIN,
        antenna=ANTENNA
    )

    # Étape 3 : Sauvegarde sur le disque dur
    if len(rx_signal) > 0:
        chemin_fichier = f"{OUTPUT_DIR}/Donnees_Brutes_IQ.npz"
        print(f"\n[Étape 3] Sauvegarde des {len(rx_signal)} échantillons IQ...")
        np.savez(chemin_fichier, ampl=rx_signal, t_start=t_start)
        print(f"✅ Fichier sauvegardé avec succès dans : {chemin_fichier}")
        print(f"Taille finale sur le disque : {os.path.getsize(chemin_fichier)/1e6:.2f} Mo")
    else:
        print("❌ Échec : Aucun échantillon n'a pu être capturé.")

if __name__ == "__main__":
    main()