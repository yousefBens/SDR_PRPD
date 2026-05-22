import numpy as np
import uhd
import time
from scipy.signal import decimate, butter, lfilter
from scipy.io import wavfile
import os

# ====================================================================
# CONFIGURATION
# ====================================================================

# Fréquence de la station de radio FM (en Hertz).
# Exemple : Pour écouter une radio à 100.0 MHz, mettez 100.0e6
FM_STATION_FREQ = 100.0e6  

# Durée de l'enregistrement en secondes
RECORD_DURATION = 10.0

# Gain de réception (en dB). 40 dB à 50 dB est idéal pour la bande FM
GAIN = 40

# Nom du fichier audio de sortie
OUTPUT_AUDIO_FILE = "./Fun/Radio_FM_Enregistrement.wav"

# Paramètres SDR
RATE = 960e3        # Taux d'échantillonnage SDR: 960 kHz
AUDIO_RATE = 48000  # Taux d'échantillonnage du fichier Audio final (standard)
CHANNEL = 0
ANTENNA = "RX2"

def main():
    print("==================================================")
    print(f" DEMODULATEUR RADIO FM ({FM_STATION_FREQ / 1e6:.1f} MHz)")
    print("==================================================")
    
    print("1. Initialisation de l'USRP...")
    usrp = uhd.usrp.MultiUSRP()
    
    usrp.set_rx_rate(RATE, CHANNEL)
    usrp.set_rx_freq(uhd.types.TuneRequest(FM_STATION_FREQ), CHANNEL)
    usrp.set_rx_gain(GAIN, CHANNEL)
    usrp.set_rx_antenna(ANTENNA, CHANNEL)
    
    # Laisser le temps à l'oscillateur de se stabiliser sur la nouvelle fréquence
    time.sleep(0.5)

    num_samps = int(RATE * RECORD_DURATION)
    
    print(f"2. Enregistrement radio en cours... ({RECORD_DURATION} secondes)")
    # Réception des échantillons IQ (signaux bruts complexes)
    samples = usrp.recv_num_samps(
        num_samps,
        FM_STATION_FREQ,
        RATE,
        [CHANNEL],
        GAIN
    )
    samples = np.asarray(samples, dtype=np.complex64).ravel()
    
    print("3. Démodulation FM...")
    
    # Étape A : Filtrage passe-bas pour isoler la station FM cible
    # On garde +/- 100 kHz autour du centre pour éliminer les autres stations
    cutoff = 100e3
    b, a = butter(4, cutoff / (RATE / 2), btype='low')
    samples_filtered = lfilter(b, a, samples)
    
    # Étape B : Démodulation (différence de phase = fréquence instantanée)
    # L'angle entre un échantillon complexe et le précédent donne directement l'audio !
    fm_demod = np.angle(samples_filtered[1:] * np.conj(samples_filtered[:-1]))
    
    print("4. Traitement Audio (Décimation et Désaccentuation)...")
    
    # Étape C : Décimation vers la fréquence audio
    # On réduit le nombre de points : On passe de 960 kHz à 48 kHz (Facteur = 20)
    decimation_factor = int(RATE / AUDIO_RATE)
    audio_signal = decimate(fm_demod, decimation_factor, ftype='iir')
    
    # Étape D : Désaccentuation (De-emphasis) - Essentiel pour la qualité FM
    # En Europe, la constante de temps (tau) des émetteurs FM est de 50 microsecondes.
    # Aux USA, c'est 75 microsecondes.
    tau = 50e-6
    alpha = np.exp(-1.0 / (AUDIO_RATE * tau))
    audio_deemph = lfilter([1 - alpha], [1, -alpha], audio_signal)
    
    # Étape E : Normalisation du volume pour éviter la saturation (clipping)
    audio_norm = audio_deemph / np.max(np.abs(audio_deemph))
    
    # Conversion en 16 bits (format standard WAV)
    audio_16bit = np.int16(audio_norm * 32767)
    
    print(f"5. Sauvegarde du fichier audio : {OUTPUT_AUDIO_FILE}")
    wavfile.write(OUTPUT_AUDIO_FILE, AUDIO_RATE, audio_16bit)
    
    print("\nTerminé ! Vous pouvez écouter le fichier WAV généré.")

if __name__ == "__main__":
    main()
