#!/usr/bin/env python3

import uhd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# ============================================================
# PARAMÈTRES
# ============================================================

SERIAL = "306BD15"
CHANNEL = 0
ANTENNA = "RX2"

FREQ_HZ = 1900e6
RATE_HZ = 56e6
GAIN_DB = 76.0

DURATION_S = 0.08

# ------------------------------------------------------------
# FILTRES IF À COMPARER
# ------------------------------------------------------------

FILTER_ORDER = 4

CUTOFF_LOW_HZ = 5e6       # filtre étroit
CUTOFF_HIGH_HZ = 27e6    # filtre large

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
# ACQUISITION
# ============================================================

cmd = uhd.types.StreamCMD(
    uhd.types.StreamMode.num_done
)

cmd.num_samps = n_samples
cmd.stream_now = True

rx_streamer.issue_stream_cmd(
    cmd
)

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
# SUPPRESSION DC
# ============================================================

samples_dc = samples - np.mean(samples)

# ============================================================
# CRÉATION DES DEUX FILTRES
# ============================================================

nyquist = RATE_HZ / 2

sos_low = butter(
    FILTER_ORDER,
    CUTOFF_LOW_HZ / nyquist,
    btype="low",
    output="sos"
)

sos_high = butter(
    FILTER_ORDER,
    CUTOFF_HIGH_HZ / nyquist,
    btype="low",
    output="sos"
)

# ============================================================
# FILTRAGE DU MÊME SIGNAL
# ============================================================

filtered_low = sosfiltfilt(
    sos_low,
    samples_dc
)

filtered_high = sosfiltfilt(
    sos_high,
    samples_dc
)

# ============================================================
# ENVELOPPES
# ============================================================

env_raw = np.abs(samples_dc)

env_low = np.abs(
    filtered_low
)

env_high = np.abs(
    filtered_high
)

# ============================================================
# AXE TEMPS
# ============================================================

t = np.arange(
    len(samples)
) / RATE_HZ

# ============================================================
# INFORMATIONS MAXIMUM
# ============================================================

max_raw = np.max(env_raw)
max_low = np.max(env_low)
max_high = np.max(env_high)

idx_raw = np.argmax(env_raw)
idx_low = np.argmax(env_low)
idx_high = np.argmax(env_high)

print()
print("========== MAXIMUMS ==========")

print(
    f"Signal brut      : "
    f"{max_raw:.6f} "
    f"à {t[idx_raw] * 1e3:.6f} ms"
)

print(
    f"Filtre étroit "
    f"{CUTOFF_LOW_HZ/1e6:.1f} MHz : "
    f"{max_low:.6f} "
    f"à {t[idx_low] * 1e3:.6f} ms"
)

print(
    f"Filtre large "
    f"{CUTOFF_HIGH_HZ/1e6:.1f} MHz : "
    f"{max_high:.6f} "
    f"à {t[idx_high] * 1e3:.6f} ms"
)

# ============================================================
# AFFICHAGE GLOBAL
# ============================================================

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    t * 1e3,
    env_raw,
    label="Signal brut |IQ|",
    alpha=0.6
)

plt.plot(
    t * 1e3,
    env_low,
    label=(
        f"Filtre étroit "
        f"fc = {CUTOFF_LOW_HZ/1e6:.1f} MHz"
    )
)

plt.plot(
    t * 1e3,
    env_high,
    label=(
        f"Filtre large "
        f"fc = {CUTOFF_HIGH_HZ/1e6:.1f} MHz"
    )
)

plt.xlabel(
    "Temps (ms)"
)

plt.ylabel(
    "Amplitude |IQ|"
)

plt.title(
    f"Comparaison filtrage IF\n"
    f"Fc = {FREQ_HZ/1e6:.1f} MHz | "
    f"Fs = {RATE_HZ/1e6:.1f} MS/s | "
    f"Gain = {GAIN_DB:.0f} dB"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()

# ============================================================
# ZOOM AUTOUR DU PLUS GRAND PULSE
# ============================================================

idx_center = np.argmax(
    env_raw
)

# +/- 10 µs autour du pulse
zoom_us = 10

zoom_samples = int(
    zoom_us * 1e-6 * RATE_HZ
)

i1 = max(
    0,
    idx_center - zoom_samples
)

i2 = min(
    len(samples),
    idx_center + zoom_samples
)

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    t[i1:i2] * 1e6,
    env_raw[i1:i2],
    label="Signal brut |IQ|",
    alpha=0.7
)

plt.plot(
    t[i1:i2] * 1e6,
    env_low[i1:i2],
    label=(
        f"Filtre étroit "
        f"{CUTOFF_LOW_HZ/1e6:.1f} MHz"
    )
)

plt.plot(
    t[i1:i2] * 1e6,
    env_high[i1:i2],
    label=(
        f"Filtre large "
        f"{CUTOFF_HIGH_HZ/1e6:.1f} MHz"
    )
)

plt.scatter(
    t[idx_raw] * 1e6,
    max_raw,
    label="Max brut"
)

plt.scatter(
    t[idx_low] * 1e6,
    max_low,
    label="Max filtre étroit"
)

plt.scatter(
    t[idx_high] * 1e6,
    max_high,
    label="Max filtre large"
)

plt.xlabel(
    "Temps (µs)"
)

plt.ylabel(
    "Amplitude |IQ|"
)

plt.title(
    "Zoom autour du pulse de plus forte amplitude"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()