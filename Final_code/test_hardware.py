import uhd
import time
import numpy as np

def main():
    print("==================================================")
    print(" OUTIL DE DIAGNOSTIC - BROCHE PPS (50 Hz)")
    print("==================================================")
    print("Initialisation du SDR (cela peut prendre quelques secondes)...")
    
    usrp = uhd.usrp.MultiUSRP()
    
    print("\nConfiguration de la source de temps sur 'external'...")
    usrp.set_clock_source("internal")
    usrp.set_time_source("external")
    time.sleep(0.5)

    duration_s = 5.0
    print(f"\nÉcoute silencieuse de la broche PPS pendant {duration_s} secondes...")
    print("Veuillez patienter...")

    pps_times = []
    start_time = time.time()
    last_pps = usrp.get_time_last_pps().get_real_secs()

    # Boucle rapide pour capturer chaque changement d'état du registre temps
    while time.time() - start_time < duration_s:
        curr_pps = usrp.get_time_last_pps().get_real_secs()
        if curr_pps != last_pps:
            pps_times.append(curr_pps)
            last_pps = curr_pps

    if len(pps_times) == 0:
        print("\n[ERREUR] AUCUN signal détecté sur la broche PPS !")
        print("Vérifiez :")
        print("1. Que votre générateur/transfo 50Hz est allumé.")
        print("2. Que le câble est bien branché sur le port 'PPS/TRIG' de l'USRP.")
        print("3. Que la tension du signal est suffisante (souvent 3.3V ou 5V CMOS).")
        return

    deltas = np.diff(pps_times)
    
    # Détection et filtrage des rebonds matériels (bounces) de moins de 10 ms
    # Un cycle 50Hz dure 20 ms. Tout ce qui est inférieur à 10 ms est du bruit/rebond.
    valid_deltas = deltas[deltas > 0.01]
    bounces = len(deltas) - len(valid_deltas)

    print("\n==================================================")
    print(" RÉSULTATS DU DIAGNOSTIC")
    print("==================================================")
    print(f"Impulsions totales détectées : {len(pps_times)}")
    if bounces > 0:
        print(f"Rebonds matériels (bruit)  : {bounces} (filtrés et ignorés)")
    print(f"Vrais cycles 50 Hz lus     : {len(valid_deltas)}")

    if len(valid_deltas) == 0:
        print("\n[ERREUR] Le signal est complètement bruité, aucun intervalle valide.")
        return

    mean_delta = np.mean(valid_deltas)
    std_delta = np.std(valid_deltas)
    freq_est = 1.0 / mean_delta if mean_delta > 0 else 0

    print(f"\nIntervalle moyen estimé  : {mean_delta:.5f} secondes")
    print(f"Stabilité (Ecart-type)   : {std_delta:.6f} secondes")
    print(f"Fréquence physique réelle: {freq_est:.2f} Hz")

    print("\n--- Analyse de la qualité du signal pour le PRPD ---")
    if 49.0 <= freq_est <= 51.0:
        if std_delta < 0.001:
            print("[EXCELLENT] Le signal 50Hz est très stable et parfaitement adapté pour la synchronisation PRPD !")
        else:
            print("[AVERTISSEMENT] La fréquence est bonne, mais le signal est un peu instable (jitter). Le PRPD pourrait être légèrement flou.")
    else:
        print(f"[ATTENTION] La fréquence détectée ({freq_est:.2f} Hz) ne correspond pas au réseau 50 Hz !")
        if 0.9 <= freq_est <= 1.1:
            print("Information : Il semble que vous ayez branché une antenne GPS (1 PPS) classique.")

    print("\nDétail des 10 premiers intervalles bruts détectés :")
    for i, d in enumerate(deltas[:10]):
        if d < 0.01:
            print(f" - {d:.5f} s (Rebond / Bruit ignoré)")
        else:
            print(f" - {d:.5f} s (Cycle normal)")
            
if __name__ == "__main__":
    main()
