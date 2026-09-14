#!/usr/bin/env python3

import uhd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# ============================================================
# CONFIGURATION SDR
# ============================================================

SERIAL = "306BD15"
CHANNEL = 0
ANTENNA = "RX2"

FREQ_HZ = 100e6
RATE_HZ = 12e6
GAIN_DB = 76.0

DURATION_S = 0.02       # 20 ms par acquisition
N_ACQUISITIONS = 5

# ============================================================
# CONFIGURATION FILTRE
# ============================================================

FILTER_ORDER = 4
CUTOFF_HZ = 5.5e6

# ============================================================
# SEUIL
# ============================================================

K = 2.0

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
# STREAM
# ============================================================

stream_args = uhd.usrp.StreamArgs(
    "fc32",
    "sc16"
)

stream_args.channels = [CHANNEL]

rx_streamer = usrp.get_rx_stream(
    stream_args
)

# ============================================================
# FILTRE IF
# ============================================================

nyquist = RATE_HZ / 2

sos = butter(
    FILTER_ORDER,
    CUTOFF_HZ / nyquist,
    btype="low",
    output="sos"
)

# ============================================================
# NOMBRE D'ÉCHANTILLONS
# ============================================================

n_samples = int(
    RATE_HZ * DURATION_S
)

# ============================================================
# FONCTION ACQUISITION
# ============================================================

def acquire():

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

    rx_md = uhd.types.RXMetadata()

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

    return samples[:total]


# ============================================================
# ACQUISITIONS SUCCESSIVES
# ============================================================

all_envelopes = []
all_thresholds = []
all_noise_median = []
all_noise_std = []

for acquisition_id in range(N_ACQUISITIONS):

    print(
        f"Acquisition "
        f"{acquisition_id + 1}/{N_ACQUISITIONS}"
    )

    # --------------------------------------------------------
    # Acquisition
    # --------------------------------------------------------

    samples = acquire()

    # --------------------------------------------------------
    # Suppression DC
    # --------------------------------------------------------

    samples = samples - np.mean(samples)

    # --------------------------------------------------------
    # Filtrage IF
    # --------------------------------------------------------

    filtered = sosfiltfilt(
        sos,
        samples
    )

    # --------------------------------------------------------
    # Enveloppe
    # --------------------------------------------------------

    envelope = np.abs(
        filtered
    )

    # --------------------------------------------------------
    # Estimation bruit
    # --------------------------------------------------------

    noise_median = np.median(
        envelope
    )

    noise_std = np.std(
        envelope
    )

    # --------------------------------------------------------
    # Seuil adaptatif
    # --------------------------------------------------------

    threshold = (
        noise_median
        + K * noise_std
    )

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    all_envelopes.append(
        envelope
    )

    all_thresholds.append(
        threshold
    )

    all_noise_median.append(
        noise_median
    )

    all_noise_std.append(
        noise_std
    )

    print(
        f"  Médiane = {noise_median:.5f}"
    )

    print(
        f"  Std     = {noise_std:.5f}"
    )

    print(
        f"  Seuil   = {threshold:.5f}"
    )


# ============================================================
# AFFICHAGE
# ============================================================

plt.figure(
    figsize=(15, 8)
)

offset_time = 0

for i, envelope in enumerate(all_envelopes):

    # Axe temps de cette acquisition
    t = (
        np.arange(len(envelope))
        / RATE_HZ
    )

    # Décalage pour afficher les acquisitions à la suite
    t = t + offset_time

    # Enveloppe
    plt.plot(
        t * 1e3,
        envelope,
        label=f"Acquisition {i+1}"
    )

    # Seuil correspondant
    plt.hlines(
        all_thresholds[i],
        xmin=t[0] * 1e3,
        xmax=t[-1] * 1e3,
        linestyles="--"
    )

    # Numéro acquisition
    plt.text(
        (
            (t[0] + t[-1])
            / 2
        ) * 1e3,
        all_thresholds[i] * 1.05,
        f"Seuil = "
        f"{all_thresholds[i]:.3f}",
        horizontalalignment="center"
    )

    offset_time += DURATION_S


# ============================================================
# GRAPHE
# ============================================================

plt.xlabel(
    "Temps (ms)"
)

plt.ylabel(
    "Amplitude |IQ|"
)

plt.title(
    "Adaptation du seuil de détection "
    "entre plusieurs acquisitions\n"
    "Seuil = médiane(A) + 2σ(A)"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# RÉSUMÉ
# ============================================================

print()
print("========== RÉSUMÉ ==========")

for i in range(N_ACQUISITIONS):

    print(
        f"Acquisition {i+1} | "
        f"Médiane = {all_noise_median[i]:.5f} | "
        f"σ = {all_noise_std[i]:.5f} | "
        f"Seuil = {all_thresholds[i]:.5f}"
    )