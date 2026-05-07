import uhd
import numpy as np
import time

def sync_usrp_with_pps():
    """
    Initialise le B200 et le synchronise avec le signal 1 Hz (PPS)
    """
    print("1. Initialisation du B200...")
    # Remplacez les arguments si besoin, ex: "type=b200, serial=31B9XXX"
    usrp = uhd.usrp.MultiUSRP("type=b200")

    print("2. Configuration de l'entrée PPS (Pulse Per Second)...")
    # On indique au USRP d'écouter le port PPS pour le temps
    usrp.set_time_source("external")
    
    # Si vous n'avez pas de signal 10MHz, l'horloge reste en interne
    # usrp.set_clock_source("internal") 

    print("3. En attente de la première impulsion (1 Hz) de votre générateur...")
    time_last = usrp.get_time_last_pps().get_real_secs()
    
    # Boucle d'attente jusqu'à voir le compteur changer
    while True:
        time_curr = usrp.get_time_last_pps().get_real_secs()
        if time_curr != time_last:
            print("   -> Impulsion PPS détectée !")
            break
        time.sleep(0.001)

    print("4. Programmation du compteur à 0.0 pour la PROCHAINE impulsion...")
    # A la prochaine impulsion matérielle, le temps interne sera exactement 0.000000
    usrp.set_time_next_pps(uhd.types.TimeSpec(0.0))

    print("5. Attente de la prochaine impulsion (environ 1 seconde)...")
    time.sleep(1.5)

    current_time = usrp.get_time_now().get_real_secs()
    print(f"Synchronisation réussie ! Temps actuel du SDR : {current_time:.4f} secondes\n")
    
    return usrp

def test_prpd_sync(usrp):
    """
    Teste la réception et calcule la phase 50 Hz à partir des timestamps synchronisés.
    """
    print("--- Démarrage du test de phase PRPD ---")
    
    # Paramétrages arbitraires pour le test
    usrp.set_rx_rate(1e6, 0) # 1 MS/s
    usrp.set_rx_freq(uhd.types.TuneRequest(300e6), 0) # 300 MHz
    
    # Configuration du flux de données
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    streamer = usrp.get_rx_stream(st_args)
    
    # Ordre de lancer la réception continue
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)
    
    buffer = np.zeros(1000, dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    # Capture de quelques paquets pour vérifier les timestamps
    for i in range(10):
        # Réception d'un paquet
        num_samps = streamer.recv(buffer, metadata)
        
        if metadata.error_code == uhd.types.RXMetadataErrorCode.none:
            # Timestamp du premier échantillon du paquet (en secondes)
            t = metadata.time_spec.get_real_secs()
            
            # --- LE SECRET DU PRPD EST ICI ---
            # Puisque la seconde 0, 1, 2... correspond exactement au passage par zéro :
            time_fraction = t % 1.0  # Fraction de seconde (ex: 0.145)
            
            # Le signal secteur est à 50 Hz, soit 1 cycle toutes les 0.02 secondes (20 ms)
            # On calcule où on se trouve dans ce cycle de 20 ms
            cycle_time = time_fraction % 0.02 
            
            # On convertit ce temps en angle (0 à 360 degrés)
            phase_degrees = (cycle_time / 0.02) * 360.0
            
            print(f"Paquet reçu à t={t:.6f} s -> Phase 50Hz extrapolée : {phase_degrees:05.1f}°")
        else:
            print(f"Erreur de réception: {metadata.error_code}")
            
        # On attend un peu avant le prochain paquet pour ne pas flooder la console
        time.sleep(0.3)

if __name__ == "__main__":
    try:
        usrp_device = sync_usrp_with_pps()
        test_prpd_sync(usrp_device)
    except Exception as e:
        print(f"Erreur lors de l'exécution : {e}")
