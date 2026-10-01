#!/usr/bin/env python3
"""
core/prpd_processor.py
-----------------------
Traitement signal PRPD — adapté de PRPD_sync_RX.py.

Ajoute la conversion dBFS → dBm via la calibration.

La fréquence de coupure du filtre passe-bas est fournie
par l'IHM afin d'utiliser le même cutoff pour :
    - le scan spectral
    - le traitement PRPD
"""

from __future__ import annotations

import time

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


# ──────────────────────────────────────────────────────────────
# Paramètres PRPD
# ──────────────────────────────────────────────────────────────

PRPD_NOISE_STD_MULT: float = 2.0   # seuil = médiane + N*std
PRPD_MIN_DISTANCE_US: float = 20   # µs entre deux pulses
PRPD_FILTER_ORDER: int = 4
EPSILON: float = 1e-12


# ──────────────────────────────────────────────────────────────
# Synchronisation PPS 50 Hz
# ──────────────────────────────────────────────────────────────

def sync_on_external_50hz_pps(usrp) -> None:
    """
    Synchronise l'horloge USRP sur le front montant PPS externe
    (50 Hz injecté).

    Doit être appelé avant chaque acquisition PRPD.
    """

    usrp.set_clock_source("internal")
    usrp.set_time_source("external")

    t_last = usrp.get_time_last_pps().get_real_secs()

    while usrp.get_time_last_pps().get_real_secs() == t_last:
        time.sleep(0.001)

    usrp.set_time_next_pps(
        __import__("uhd").types.TimeSpec(0.0)
    )

    time.sleep(0.05)


# ──────────────────────────────────────────────────────────────
# Acquisition synchronisée
# ──────────────────────────────────────────────────────────────

def rx_prpd_sync(
    usrp,
    freq_hz: float,
    rate_hz: float,
    duration_s: float,
    gain_db: float,
    channel: int = 0,
    antenna: str = "RX2",
    progress_cb=None,
) -> tuple[np.ndarray, float | None]:
    """
    Acquisition PRPD synchronisée sur PPS 50 Hz.

    Retourne :
        samples   : échantillons IQ
        t_start   : temps absolu du premier échantillon

    progress_cb(pct) permet de mettre à jour
    la barre de progression.
    """

    import uhd

    # Nombre total d'échantillons à acquérir
    num_samps = int(
        duration_s * rate_hz
    )

    # ----------------------------------------------------------
    # Configuration USRP
    # ----------------------------------------------------------

    usrp.set_rx_rate(
        rate_hz,
        channel
    )

    usrp.set_rx_freq(
        uhd.types.TuneRequest(freq_hz),
        channel
    )

    usrp.set_rx_gain(
        gain_db,
        channel
    )

    usrp.set_rx_antenna(
        antenna,
        channel
    )

    usrp.set_rx_dc_offset(
        True,
        channel
    )

    usrp.set_rx_iq_balance(
        True,
        channel
    )

    # Temps de stabilisation
    time.sleep(0.5)

    # ----------------------------------------------------------
    # Création du streamer
    # ----------------------------------------------------------

    stream_args = uhd.usrp.StreamArgs(
        "fc32",
        "sc16"
    )

    stream_args.channels = [
        channel
    ]

    rx_streamer = usrp.get_rx_stream(
        stream_args
    )

    rx_md = uhd.types.RXMetadata()

    received_buf = np.zeros(
        num_samps,
        dtype=np.complex64
    )

    # ----------------------------------------------------------
    # Début acquisition aligné sur une période 50 Hz
    #
    # 50 Hz -> période = 20 ms
    # ----------------------------------------------------------

    current_t = (
        usrp.get_time_now()
        .get_real_secs()
    )

    future_t = current_t + 0.1

    start_t = (
        np.ceil(
            future_t / 0.02
        )
        * 0.02
    )

    # ----------------------------------------------------------
    # Commande d'acquisition
    # ----------------------------------------------------------

    cmd = uhd.types.StreamCMD(
        uhd.types.StreamMode.num_done
    )

    cmd.num_samps = num_samps
    cmd.stream_now = False
    cmd.time_spec = uhd.types.TimeSpec(
        start_t
    )

    rx_streamer.issue_stream_cmd(
        cmd
    )

    # ----------------------------------------------------------
    # Réception IQ
    # ----------------------------------------------------------

    buff = np.zeros(
        (1, 4096),
        dtype=np.complex64
    )

    total = 0
    t_start = None

    while total < num_samps:

        n = rx_streamer.recv(
            buff,
            rx_md,
            timeout=5.0
        )

        if (
            rx_md.error_code
            != uhd.types.RXMetadataErrorCode.none
        ):

            if (
                rx_md.error_code
                == uhd.types.RXMetadataErrorCode.late_command
            ):
                break

            continue

        if n > 0:

            # Temps du premier échantillon reçu
            if t_start is None:

                t_start = (
                    rx_md.time_spec
                    .get_real_secs()
                )

            end = min(
                total + n,
                num_samps
            )

            received_buf[
                total:end
            ] = buff[
                0,
                :end - total
            ]

            total = end

            # Progression
            if progress_cb is not None:

                progress_cb(
                    int(
                        total
                        / num_samps
                        * 100
                    )
                )

    return (
        received_buf[:total],
        t_start
    )


# ──────────────────────────────────────────────────────────────
# Traitement signal PRPD
# ──────────────────────────────────────────────────────────────

def process_pd_signal(
    samples: np.ndarray,
    rate_hz: float,
    t_start: float | None,
    f_offset_hz: float = 0.0,
    gain_calibration: dict | None = None,
    freq_hz: float | None = None,
    cutoff_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Détecte les pulses PD dans les samples
    et calcule leurs phases PRPD.

    Paramètres
    ----------
    samples :
        Échantillons IQ complex64.

    rate_hz :
        Sampling rate / débit I/Q.

    t_start :
        Temps absolu du premier échantillon.

    f_offset_hz :
        Décalage fréquentiel éventuel permettant
        de recentrer le signal.

    gain_calibration :
        Calibration utilisée pour convertir
        les amplitudes dBFS en dBm.

    freq_hz :
        Fréquence centrale RX utilisée pour
        la calibration.

    cutoff_hz :
        Fréquence de coupure du filtre passe-bas.
        Cette valeur vient de l'IHM et est la même
        que celle utilisée pour le scan spectral.

    Retourne
    --------
    phases_deg :
        Phase 50 Hz de chaque pulse, entre 0 et 360°.

    amplitudes :
        Amplitude des pulses en dBm si calibration
        disponible, sinon en dBFS.

    info :
        Informations de diagnostic.
    """

    # ----------------------------------------------------------
    # Préparation des IQ
    # ----------------------------------------------------------

    samples = np.asarray(
        samples,
        dtype=np.complex64
    ).ravel()

    N = len(samples)

    if N == 0 or t_start is None:

        return (
            np.array([]),
            np.array([]),
            {
                "n_pulses": 0,
                "unit": "dBFS"
            }
        )

    # ==========================================================
    # 1. Suppression de la composante DC
    # ==========================================================

    samples = (
        samples
        - np.mean(samples)
    )

    # Axe temporel
    t = (
        np.arange(N)
        / rate_hz
    )

    # ==========================================================
    # 2. Translation fréquentielle vers DC
    # ==========================================================

    if abs(f_offset_hz) > 1.0:

        samples = (
            samples
            * np.exp(
                -1j
                * 2
                * np.pi
                * f_offset_hz
                * t
            )
        )

    # ==========================================================
    # 3. Filtre passe-bas
    #
    # Le cutoff vient maintenant de l'IHM.
    # ==========================================================

    if cutoff_hz is None:

        raise ValueError(
            "cutoff_hz doit être fourni "
            "pour le traitement PRPD."
        )

    # Fréquence de Nyquist
    nyq = (
        rate_hz
        / 2.0
    )

    # ----------------------------------------------------------
    # Protection :
    #
    # Le cutoff ne doit jamais être >= Fs/2.
    #
    # Exemple :
    # Fs = 12 MSps
    # Nyquist = 6 MHz
    #
    # Si l'utilisateur demande 10 MHz,
    # le cutoff réel sera limité à :
    #
    # 0.95 * 6 MHz = 5.7 MHz
    # ----------------------------------------------------------

    actual_cutoff_hz = min(
        float(cutoff_hz),
        nyq * 0.95
    )

    # Protection supplémentaire
    if actual_cutoff_hz <= 0:

        raise ValueError(
            "La fréquence de coupure doit "
            "être strictement positive."
        )

    # ----------------------------------------------------------
    # Création filtre Butterworth
    # ----------------------------------------------------------

    sos = butter(
        PRPD_FILTER_ORDER,
        actual_cutoff_hz / nyq,
        btype="low",
        output="sos",
    )

    # ----------------------------------------------------------
    # Application du filtre aux IQ
    # ----------------------------------------------------------

    filtered = sosfiltfilt(
        sos,
        samples
    )

    # ==========================================================
    # 4. Calcul de l'enveloppe
    # ==========================================================

    envelope = np.abs(
        filtered
    )

    # ==========================================================
    # 5. Calcul du seuil dynamique
    #
    # threshold =
    # median(envelope)
    # + N * std(envelope)
    # ==========================================================

    noise_level = np.median(
        envelope
    )

    noise_std = np.std(
        envelope
    )

    threshold = (
        noise_level
        + PRPD_NOISE_STD_MULT
        * noise_std
    )

    # ==========================================================
    # 6. Distance minimale entre deux pulses
    # ==========================================================

    min_dist = int(
        PRPD_MIN_DISTANCE_US
        * 1e-6
        * rate_hz
    )

    # ==========================================================
    # 7. Détection des pulses
    # ==========================================================

    peaks, _ = find_peaks(
        envelope,
        height=threshold,
        distance=max(
            1,
            min_dist
        ),
    )

    # ==========================================================
    # Aucun pulse détecté
    # ==========================================================

    if len(peaks) == 0:

        return (
            np.array([]),
            np.array([]),
            {
                "n_pulses": 0,
                "unit": "dBFS",
                "threshold":
                    float(threshold),
                "noise_med":
                    float(noise_level),
                "noise_std":
                    float(noise_std),
                "cutoff_hz":
                    float(actual_cutoff_hz),
                "sample_rate_hz":
                    float(rate_hz),
            },
        )

    # ==========================================================
    # 8. Temps absolu des pulses
    # ==========================================================

    t_peaks = (
        t_start
        + peaks / rate_hz
    )

    # ==========================================================
    # 9. Phase réseau 50 Hz
    #
    # 50 Hz -> période = 20 ms
    #
    # 0 ms  ->   0°
    # 5 ms  ->  90°
    # 10 ms -> 180°
    # 15 ms -> 270°
    # 20 ms -> 360° = 0°
    # ==========================================================

    phases_deg = (
        (
            t_peaks
            % 0.02
        )
        / 0.02
    ) * 360.0

    # ==========================================================
    # 10. Amplitude des pulses
    # ==========================================================

    amps_raw = (
        envelope[peaks]
    )

    # ----------------------------------------------------------
    # Conversion amplitude -> dBFS
    # ----------------------------------------------------------

    amps_dbfs = (
        20.0
        * np.log10(
            np.clip(
                amps_raw,
                EPSILON,
                None
            )
        )
    )

    # Par défaut, sortie en dBFS
    unit = "dBFS"

    amps_out = amps_dbfs

    # ==========================================================
    # 11. Conversion dBFS -> dBm
    # ==========================================================

    if (
        gain_calibration is not None
        and freq_hz is not None
    ):

        try:

            from core.calibration import (
                interpolate_reference_dbm
            )

            ref_dbm = (
                interpolate_reference_dbm(
                    gain_calibration,
                    freq_hz
                )
            )

            amps_out = (
                amps_dbfs
                + ref_dbm
            )

            unit = "dBm"

        except Exception:

            # En cas de problème avec la calibration,
            # on conserve les valeurs en dBFS.
            pass

    # ==========================================================
    # 12. Informations de diagnostic
    # ==========================================================

    info = {

        "n_pulses":
            len(peaks),

        "unit":
            unit,

        "threshold":
            float(threshold),

        "noise_med":
            float(noise_level),

        "noise_std":
            float(noise_std),

        # Cutoff réellement appliqué
        "cutoff_hz":
            float(actual_cutoff_hz),

        # Sampling rate réellement utilisé
        "sample_rate_hz":
            float(rate_hz),
    }

    # ==========================================================
    # Résultat final
    # ==========================================================

    return (
        phases_deg,
        amps_out,
        info
    )