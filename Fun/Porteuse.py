import numpy as np
import uhd
import time

# =========================
# CONFIG TX USRP
# =========================
TX_CENTER_FREQ = 200e6      # centre USRP
RATE = 3e6                # compatible RTL-SDR
TX_GAIN = 60              # commence à 10-20 dB avec antenne
CHANNEL = 0
ANTENNA = "TX/RX"

TONE_OFFSET = 1e6         # pic réel = 433.5 MHz
AMPLITUDE = 1.5
N = 4096

# =========================
# INIT USRP
# =========================
usrp = uhd.usrp.MultiUSRP()
usrp.set_tx_rate(RATE, CHANNEL)
usrp.set_tx_freq(uhd.types.TuneRequest(TX_CENTER_FREQ), CHANNEL)
usrp.set_tx_gain(TX_GAIN, CHANNEL)
usrp.set_tx_antenna(ANTENNA, CHANNEL)

print("TX rate:", usrp.get_tx_rate(CHANNEL))
print("TX center:", usrp.get_tx_freq(CHANNEL) / 1e6, "MHz")
print("TX gain:", usrp.get_tx_gain(CHANNEL))

# =========================
# SIGNAL CONTINU
# =========================
t = np.arange(N) / RATE
tone = AMPLITUDE * np.exp(1j * 2 * np.pi * TONE_OFFSET * t)
tone = tone.astype(np.complex64)

# =========================
# STREAMER
# =========================
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [CHANNEL]
tx_streamer = usrp.get_tx_stream(stream_args)

md = uhd.types.TXMetadata()
md.start_of_burst = True
md.end_of_burst = False
md.has_time_spec = False

print("===================================")
print("ÉMISSION CONTINUE")
print("Centre USRP :", TX_CENTER_FREQ / 1e6, "MHz")
print("Pic RF      :", (TX_CENTER_FREQ + TONE_OFFSET) / 1e6, "MHz")
print("RTL center  :", TX_CENTER_FREQ / 1e6, "MHz")
print("Dans le RTL : pic à +500 kHz")
print("Ctrl+C pour arrêter")
print("===================================")

try:
    while True:
        tx_streamer.send(tone, md)
        md.start_of_burst = False

except KeyboardInterrupt:
    md.end_of_burst = True
    tx_streamer.send(np.zeros(1024, dtype=np.complex64), md)
    print("Arrêt TX propre")