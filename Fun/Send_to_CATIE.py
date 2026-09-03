import numpy as np
import uhd

# =========================
# CONFIGURATION
# =========================

TX_FREQ = 70e6      # Fréquence RF : 200 MHz
RATE = 3e6           # Fréquence d'échantillonnage : 3 MS/s
TX_GAIN = 40         # Gain TX en dB
AMPLITUDE = 0.5      # Amplitude numérique (0 à 1)

CHANNEL = 0
N = 4096


# =========================
# INITIALISATION USRP
# =========================

usrp = uhd.usrp.MultiUSRP()

usrp.set_tx_rate(RATE, CHANNEL)
usrp.set_tx_freq(TX_FREQ, CHANNEL)
usrp.set_tx_gain(TX_GAIN, CHANNEL)
usrp.set_tx_antenna("TX/RX", CHANNEL)


print("Fréquence TX :", usrp.get_tx_freq(CHANNEL) / 1e6, "MHz")
print("Rate         :", usrp.get_tx_rate(CHANNEL) / 1e6, "MS/s")
print("Gain TX      :", usrp.get_tx_gain(CHANNEL), "dB")


# =========================
# SIGNAL
# =========================

# Signal constant complexe
# Cela correspond à une porteuse à la fréquence TX_FREQ
signal = AMPLITUDE * np.ones(N, dtype=np.complex64)


# =========================
# STREAM TX
# =========================

stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [CHANNEL]

tx_streamer = usrp.get_tx_stream(stream_args)

md = uhd.types.TXMetadata()
md.start_of_burst = True
md.end_of_burst = False
md.has_time_spec = False


# =========================
# EMISSION CONTINUE
# =========================

print("\nEmission continue à", TX_FREQ / 1e6, "MHz")
print("Ctrl+C pour arrêter\n")

try:

    while True:

        tx_streamer.send(signal, md)

        md.start_of_burst = False


except KeyboardInterrupt:

    print("\nArrêt TX")

    md.end_of_burst = True

    tx_streamer.send(
        np.zeros(N, dtype=np.complex64),
        md
    )