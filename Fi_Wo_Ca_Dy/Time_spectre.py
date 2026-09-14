#!/usr/bin/env python3

import uhd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ============================================================
# PARAMÈTRES
# ============================================================

SERIAL = "306BD15"
CHANNEL = 0
ANTENNA = "RX2"

FREQ_HZ = 100e6       # 1.2 GHz
RATE_HZ = 56e6        # 12 MS/s
GAIN_DB = 40.0        # gain fixe

DURATION_S = 0.08     # 10 ms

sin_A = 0.8

# ============================================================
# CONNEXION USRP
# ============================================================

usrp = uhd.usrp.MultiUSRP(
    f"serial={SERIAL}"
)

usrp.set_rx_rate(
    RATE_HZ,
    CHANNEL
)

usrp.set_rx_antenna(
    ANTENNA,
    CHANNEL
)

usrp.set_rx_gain(
    GAIN_DB,
    CHANNEL
)

usrp.set_rx_freq(
    uhd.types.TuneRequest(FREQ_HZ),
    CHANNEL
)


# ============================================================
# NOMBRE D'ÉCHANTILLONS
# ============================================================

n_samples = int(
    RATE_HZ * DURATION_S
)

print(f"Fréquence : {FREQ_HZ / 1e6:.1f} MHz")
print(f"Gain      : {GAIN_DB:.1f} dB")
print(f"Rate      : {RATE_HZ / 1e6:.1f} MS/s")
print(f"Durée     : {DURATION_S * 1000:.1f} ms")
print(f"Samples   : {n_samples}")


# ============================================================
# STREAM RX
# ============================================================

stream_args = uhd.usrp.StreamArgs(
    "fc32",
    "sc16"
)

stream_args.channels = [
    CHANNEL
]

rx_streamer = usrp.get_rx_stream(
    stream_args
)

rx_md = uhd.types.RXMetadata()


# ============================================================
# BUFFER
# ============================================================

samples = np.zeros(
    n_samples,
    dtype=np.complex64
)

buffer = np.zeros(
    (
        1,
        rx_streamer.get_max_num_samps()
    ),
    dtype=np.complex64
)


# ============================================================
# LANCEMENT ACQUISITION
# ============================================================

cmd = uhd.types.StreamCMD(
    uhd.types.StreamMode.num_done
)

cmd.num_samps = n_samples
cmd.stream_now = True

rx_streamer.issue_stream_cmd(
    cmd
)


# ============================================================
# RÉCEPTION
# ============================================================

total = 0

while total < n_samples:

    n = rx_streamer.recv(
        buffer,
        rx_md,
        timeout=3.0
    )

    if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
        raise RuntimeError(
            rx_md.strerror()
        )

    if n <= 0:
        break

    count = min(
        n,
        n_samples - total
    )

    samples[
        total:total + count
    ] = buffer[
        0,
        :count
    ]

    total += count


samples = samples[:total]

print(
    f"Échantillons reçus : {len(samples)}"
)

# ============================================================
# SAUVEGARDE DES ÉCHANTILLONS
# ============================================================

SAVE_DIR = Path("/home/yousef/Documents/testing_scripts/GEVernova/Fi_Wo_Ca_Dy")
SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

filename = (
    f"SDR_"
    f"{timestamp}_"
    f"Serial-{SERIAL}_"
    f"Ch-{CHANNEL}_"
    f"Ant-{ANTENNA}_"
    f"Fc-{FREQ_HZ/1e6:.1f}MHz_"
    f"Fs-{RATE_HZ/1e6:.1f}MSps_"
    f"Gain-{GAIN_DB:.1f}dB_"
    f"Dur-{DURATION_S*1000:.1f}ms_"
    f"N-{len(samples)}"
    f".npz"
)

save_path = (
    SAVE_DIR
    / filename
)

np.savez(
    save_path,

    # Données
    samples=samples,

    # Metadata
    serial=SERIAL,
    channel=CHANNEL,
    antenna=ANTENNA,
    frequency_hz=FREQ_HZ,
    rate_hz=RATE_HZ,
    gain_db=GAIN_DB,
    duration_s=DURATION_S,
    n_samples=len(samples),
    timestamp=timestamp,
)

print()
print(
    f"Acquisition sauvegardée :\n"
    f"{save_path}"
)

# ============================================================
# AXE TEMPS
# ============================================================

t = np.arange(
    len(samples)
) / RATE_HZ


signal_50hz = sin_A*np.sin(
    2 * np.pi * 50 * t
)


# ============================================================
# AFFICHAGE
# ============================================================
print(samples.real)

plt.figure(figsize=(14, 6))

# Signal SDR
# plt.plot(
#     t * 1e3,
#     np.abs(samples),
#     label="Signal SDR |IQ|"
# )

plt.plot(
    t * 1e3,
    samples.imag,
    label="Signal SDR Imag"
)
plt.plot(
    t * 1e3,
    samples.real,
    label="Signal SDR Real"
)

# Signal 50 Hz
plt.plot(
    t * 1e3,
    signal_50hz,
    label="Signal 50 Hz"
)

plt.xlabel("Temps (ms)")
plt.ylabel("Amplitude")

plt.title(
    f"Signal temporel - "
    f"{FREQ_HZ / 1e6:.1f} MHz - "
    f"Gain {GAIN_DB:.0f} dB"
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################


plt.figure(figsize=(14, 6))


plt.plot(
    t * 1e3,
    np.abs(samples),
    label="Signal SDR |IQ|"
)

# Signal 50 Hz
plt.plot(
    t * 1e3,
    signal_50hz,
    label="Signal 50 Hz"
)


plt.xlabel("Temps (ms)")
plt.ylabel("Amplitude")

plt.title(
    f"Signal temporel - "
    f"{FREQ_HZ / 1e6:.1f} MHz - "
    f"Gain {GAIN_DB:.0f} dB"
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()