#!/usr/bin/env python3

import uhd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, find_peaks


# ============================================================
# CONFIGURATION SDR
# ============================================================

SERIAL = "306BD15"
CHANNEL = 0
ANTENNA = "RX2"

FREQ_HZ = 100e6
RATE_HZ = 12e6
GAIN_DB = 76.0

DURATION_S = 0.02      # 20 ms


# ============================================================
# FILTRE IF
# ============================================================

FILTER_ORDER = 4
CUTOFF_HZ = 5.5e6


# ============================================================
# DÉTECTION
# ============================================================

K = 2.0

# Comparaison de plusieurs seuils temporels
DISTANCES_US = [
    1,
    20,
    100
]


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
# ACQUISITION
# ============================================================

n_samples = int(
    RATE_HZ * DURATION_S
)

stream_args = uhd.usrp.StreamArgs(
    "fc32",
    "sc16"
)

stream_args.channels = [CHANNEL]

rx_streamer = usrp.get_rx_stream(
    stream_args
)

rx_md = uhd.types.RXMetadata()

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

    if rx_md.error_code != \
            uhd.types.RXMetadataErrorCode.none:

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


# ============================================================
# SUPPRESSION DC
# ============================================================

samples = samples - np.mean(samples)


# ============================================================
# FILTRAGE IF
# ============================================================

nyquist = RATE_HZ / 2

sos = butter(
    FILTER_ORDER,
    CUTOFF_HZ / nyquist,
    btype="low",
    output="sos"
)

filtered = sosfiltfilt(
    sos,
    samples
)


# ============================================================
# ENVELOPPE
# ============================================================

envelope = np.abs(
    filtered
)


# ============================================================
# SEUIL D'AMPLITUDE
# ============================================================

noise_level = np.median(
    envelope
)

noise_std = np.std(
    envelope
)

threshold = (
    noise_level
    + K * noise_std
)

print()
print(
    f"Médiane bruit : {noise_level:.6f}"
)

print(
    f"Écart-type    : {noise_std:.6f}"
)

print(
    f"Seuil         : {threshold:.6f}"
)


# ============================================================
# AXE TEMPS
# ============================================================

t = np.arange(
    len(envelope)
) / RATE_HZ

t_us = t * 1e6


# ============================================================
# DÉTECTION POUR PLUSIEURS DISTANCES
# ============================================================

detections = {}

for distance_us in DISTANCES_US:

    min_distance_samples = int(
        distance_us
        * 1e-6
        * RATE_HZ
    )

    peaks, properties = find_peaks(
        envelope,
        height=threshold,
        distance=max(
            1,
            min_distance_samples
        )
    )

    detections[
        distance_us
    ] = peaks

    print(
        f"Distance = {distance_us:3d} µs "
        f"→ {len(peaks)} pics détectés"
    )


# ============================================================
# AFFICHAGE
# ============================================================

fig, axes = plt.subplots(
    len(DISTANCES_US),
    1,
    figsize=(15, 10),
    sharex=True
)

for ax, distance_us in zip(
        axes,
        DISTANCES_US
):

    peaks = detections[
        distance_us
    ]

    # Enveloppe
    ax.plot(
        t_us,
        envelope,
        label="Enveloppe"
    )

    # Seuil amplitude
    ax.axhline(
        threshold,
        linestyle="--",
        label="Seuil d'amplitude"
    )

    # Pics détectés
    ax.scatter(
        t_us[peaks],
        envelope[peaks],
        marker="x",
        s=50,
        label=(
            f"Pics détectés : "
            f"{len(peaks)}"
        )
    )

    ax.set_ylabel(
        "|IQ|"
    )

    ax.set_title(
        f"Distance temporelle minimale = "
        f"{distance_us} µs"
    )

    ax.grid(True)

    ax.legend()


axes[-1].set_xlabel(
    "Temps (µs)"
)

fig.suptitle(
    "Influence du seuil temporel "
    "sur la détection des impulsions"
)

plt.tight_layout()

plt.show()