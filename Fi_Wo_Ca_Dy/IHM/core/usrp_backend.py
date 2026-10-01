#!/usr/bin/env python3
"""
core/usrp_backend.py
--------------------
Backend USRP pour :
  - Scan spectral avec gain fixe
  - Acquisition PRPD

Le même paramètre lowpass_cutoff_hz est utilisé pour :
  - le scan spectral
  - le traitement PRPD
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.calibration import (
    amplitude_to_dbfs,
    dbfs_to_dbm,
)

from core.prpd_processor import (
    process_pd_signal,
    rx_prpd_sync,
    sync_on_external_50hz_pps,
)


# =============================================================
# Résultat d'une fréquence
# =============================================================

@dataclass
class FreqResult:
    freq_mhz: float
    freq_hz: float
    gain_db: float

    max_dbfs: float
    robust_dbfs: float
    median_dbfs: float

    clipping_fraction: float

    max_dbm: float
    robust_dbm: float
    ref_dbm: float

    status: str


# =============================================================
# Paramètres par défaut
# =============================================================

DEFAULT_PARAMS = {
    "usrp_serial":       "306BD15",
    "channel":           0,
    "antenna":           "RX2",

    # ---------------------------------------------------------
    # Scan spectral
    # ---------------------------------------------------------

    "f_start_hz":        100e6,
    "f_stop_hz":         2.0e9,
    "step_hz":           12e6,

    # Sampling rate commun au scan et au PRPD
    "rate_hz":           56e6,

    # Gain FIXE pour le spectre
    "scan_gain_db":      40.0,

    "lo_offset_hz":      1e6,

    "discard_s":         0.001,
    "useful_s":          0.021,
    "settling_s":        0.100,

    # ---------------------------------------------------------
    # Traitement commun
    #
    # Ce cutoff est maintenant utilisé par :
    #   1. Scan spectral
    #   2. PRPD
    # ---------------------------------------------------------

    "lowpass_cutoff_hz": 27.99e6,

    "filter_order":      4,
    "robust_percentile": 99.99,
    "filter_edge_s":     0.0002,
    "clipping_amp":      0.98,

    # ---------------------------------------------------------
    # PRPD
    # ---------------------------------------------------------

    "prpd_freq_hz":      1.196e9,
    "prpd_gain_db":      40.0,
    "prpd_duration_s":   1.0,
    "prpd_f_offset_hz":  0.0,
    "prpd_n_acq":        1,
}


# =============================================================
# SCAN SPECTRAL
# =============================================================

class ScanThread(QThread):

    result_ready = pyqtSignal(object)
    progress = pyqtSignal(int)
    scan_done = pyqtSignal(list)
    log_message = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        params: dict,
        cal_calibrations: dict,
    ) -> None:

        super().__init__()

        self.params = {
            **DEFAULT_PARAMS,
            **params,
        }

        self.cal_calibrations = cal_calibrations
        self._stop_flag = False

    # ---------------------------------------------------------
    # Stop
    # ---------------------------------------------------------

    def stop(self) -> None:
        self._stop_flag = True

    # ---------------------------------------------------------
    # Log
    # ---------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.log_message.emit(msg)

    # ---------------------------------------------------------
    # Acquisition
    # ---------------------------------------------------------

    def _acquire(
        self,
        usrp,
        n_samples: int,
    ) -> np.ndarray:

        import uhd

        p = self.params

        stream_args = uhd.usrp.StreamArgs(
            "fc32",
            "sc16",
        )

        stream_args.channels = [
            p["channel"]
        ]

        rx_streamer = usrp.get_rx_stream(
            stream_args
        )

        rx_md = uhd.types.RXMetadata()

        samples = np.zeros(
            n_samples,
            dtype=np.complex64,
        )

        buff = np.zeros(
            (
                1,
                rx_streamer.get_max_num_samps(),
            ),
            dtype=np.complex64,
        )

        cmd = uhd.types.StreamCMD(
            uhd.types.StreamMode.num_done
        )

        cmd.num_samps = n_samples
        cmd.stream_now = True

        rx_streamer.issue_stream_cmd(cmd)

        total = 0

        while total < n_samples:

            n = rx_streamer.recv(
                buff,
                rx_md,
                timeout=3.0,
            )

            if (
                rx_md.error_code
                != uhd.types.RXMetadataErrorCode.none
            ):
                raise RuntimeError(
                    f"UHD RX error: {rx_md.strerror()}"
                )

            if n <= 0:
                raise RuntimeError(
                    "Aucun échantillon reçu."
                )

            count = min(
                n,
                n_samples - total,
            )

            samples[
                total:
                total + count
            ] = buff[0, :count]

            total += count

        # -----------------------------------------------------
        # Suppression du début de l'acquisition
        # -----------------------------------------------------

        discard = int(
            p["discard_s"]
            * p["rate_hz"]
        )

        return samples[discard:]

    # ---------------------------------------------------------
    # Métriques scan
    # ---------------------------------------------------------

    def _compute_metrics(
        self,
        samples: np.ndarray,
        sos,
    ) -> dict:

        from scipy.signal import sosfiltfilt

        p = self.params

        # -----------------------------------------------------
        # Application du filtre passe-bas
        #
        # Le filtre est construit dans run() avec
        # p["lowpass_cutoff_hz"].
        # -----------------------------------------------------

        filtered = sosfiltfilt(
            sos,
            samples,
        )

        # -----------------------------------------------------
        # Suppression des bords du filtre
        # -----------------------------------------------------

        edge = int(
            p["filter_edge_s"]
            * p["rate_hz"]
        )

        if (
            edge > 0
            and filtered.size > 2 * edge
        ):
            filtered = filtered[
                edge:
                -edge
            ]

        # -----------------------------------------------------
        # Enveloppe IQ
        # -----------------------------------------------------

        env = np.abs(filtered)

        # -----------------------------------------------------
        # Calcul métriques
        # -----------------------------------------------------

        return {

            "max_dbfs":
                amplitude_to_dbfs(
                    float(
                        np.max(env)
                    )
                ),

            "robust_dbfs":
                amplitude_to_dbfs(
                    float(
                        np.percentile(
                            env,
                            p["robust_percentile"],
                        )
                    )
                ),

            "median_dbfs":
                amplitude_to_dbfs(
                    float(
                        np.median(env)
                    )
                ),

            "clipping_fraction":
                float(
                    np.mean(
                        np.abs(samples)
                        >= p["clipping_amp"]
                    )
                ),
        }

    # ---------------------------------------------------------
    # Configuration USRP
    # ---------------------------------------------------------

    def _tune(
        self,
        usrp,
        freq_hz: float,
        gain_db: float,
    ) -> tuple[float, float]:

        import uhd

        p = self.params

        # Sampling rate
        usrp.set_rx_rate(
            p["rate_hz"],
            p["channel"],
        )

        # Antenne
        usrp.set_rx_antenna(
            p["antenna"],
            p["channel"],
        )

        # Gain FIXE
        usrp.set_rx_gain(
            gain_db,
            p["channel"],
        )

        # Tuning
        tune_req = uhd.types.TuneRequest(
            freq_hz,
            p["lo_offset_hz"],
        )

        usrp.set_rx_freq(
            tune_req,
            p["channel"],
        )

        # Stabilisation
        time.sleep(
            p["settling_s"]
        )

        actual_freq = float(
            usrp.get_rx_freq(
                p["channel"]
            )
        )

        actual_gain = float(
            usrp.get_rx_gain(
                p["channel"]
            )
        )

        return (
            actual_freq,
            actual_gain
        )

    # ---------------------------------------------------------
    # Calibration correspondant au gain
    # ---------------------------------------------------------

    def _get_calibration(
        self,
        gain_db: float,
    ) -> dict:

        if not self.cal_calibrations:
            raise RuntimeError(
                "Aucune calibration disponible."
            )

        gains = np.asarray(
            sorted(
                float(g)
                for g
                in self.cal_calibrations.keys()
            )
        )

        selected_gain = float(
            gains[
                np.argmin(
                    np.abs(
                        gains - gain_db
                    )
                )
            ]
        )

        return self.cal_calibrations[
            selected_gain
        ]

    # ---------------------------------------------------------
    # Scan
    # ---------------------------------------------------------

    def run(self) -> None:

        import uhd
        from scipy.signal import butter

        p = self.params

        fixed_gain = float(
            p["scan_gain_db"]
        )

        # =====================================================
        # FILTRE SCAN
        #
        # Utilise directement le cutoff choisi dans l'IHM.
        # =====================================================

        nyq = (
            p["rate_hz"]
            / 2.0
        )

        requested_cutoff = float(
            p["lowpass_cutoff_hz"]
        )

        if requested_cutoff <= 0:

            self.error.emit(
                "Low-pass cutoff invalide : "
                "la valeur doit être > 0."
            )

            return

        # Protection Nyquist
        cutoff = min(
            requested_cutoff,
            nyq * 0.9999,
        )

        sos = butter(
            p["filter_order"],
            cutoff / nyq,
            btype="low",
            output="sos",
        )

        self._log(
            f"Filtre scan : "
            f"Fc={cutoff/1e6:.3f} MHz | "
            f"Fs={p['rate_hz']/1e6:.3f} MS/s"
        )

        # =====================================================
        # Nombre de samples
        # =====================================================

        n_total = int(
            (
                p["discard_s"]
                + p["useful_s"]
            )
            * p["rate_hz"]
        )

        # =====================================================
        # Fréquences du scan
        # =====================================================

        freqs_hz = np.arange(
            p["f_start_hz"],
            p["f_stop_hz"]
            + p["step_hz"] / 2,
            p["step_hz"],
        )

        n_freqs = len(
            freqs_hz
        )

        # =====================================================
        # Connexion USRP
        # =====================================================

        try:

            serial_arg = (
                f"serial={p['usrp_serial']}"
                if p["usrp_serial"]
                else ""
            )

            self._log(
                "Connexion USRP..."
            )

            usrp = uhd.usrp.MultiUSRP(
                serial_arg
            )

        except Exception as exc:

            self.error.emit(
                f"Connexion USRP impossible :\n{exc}"
            )

            return

        self._log(
            f"Scan : {n_freqs} fréquences | "
            f"Gain fixe = {fixed_gain:.0f} dB | "
            f"Fs = {p['rate_hz']/1e6:.2f} MS/s | "
            f"Cutoff = {cutoff/1e6:.2f} MHz"
        )

        results = []

        # =====================================================
        # Boucle scan
        # =====================================================

        for idx, freq_hz in enumerate(
            freqs_hz,
            start=1,
        ):

            if self._stop_flag:
                break

            try:

                # ---------------------------------------------
                # Configuration fréquence + gain
                # ---------------------------------------------

                actual_freq, actual_gain = (
                    self._tune(
                        usrp,
                        float(freq_hz),
                        fixed_gain,
                    )
                )

                # ---------------------------------------------
                # Acquisition
                # ---------------------------------------------

                samples = self._acquire(
                    usrp,
                    n_total,
                )

                # ---------------------------------------------
                # Métriques
                #
                # Les samples passent par le LPF construit
                # avec le cutoff de l'IHM.
                # ---------------------------------------------

                metrics = self._compute_metrics(
                    samples,
                    sos,
                )

                # ---------------------------------------------
                # Calibration
                # ---------------------------------------------

                cal = self._get_calibration(
                    actual_gain
                )

                max_dbm, ref_dbm = dbfs_to_dbm(
                    cal,
                    metrics["max_dbfs"],
                    actual_freq,
                )

                robust_dbm, _ = dbfs_to_dbm(
                    cal,
                    metrics["robust_dbfs"],
                    actual_freq,
                )

                # ---------------------------------------------
                # Résultat
                # ---------------------------------------------

                result = FreqResult(

                    freq_mhz=
                        actual_freq / 1e6,

                    freq_hz=
                        actual_freq,

                    gain_db=
                        actual_gain,

                    max_dbfs=
                        metrics["max_dbfs"],

                    robust_dbfs=
                        metrics["robust_dbfs"],

                    median_dbfs=
                        metrics["median_dbfs"],

                    clipping_fraction=
                        metrics[
                            "clipping_fraction"
                        ],

                    max_dbm=
                        max_dbm,

                    robust_dbm=
                        robust_dbm,

                    ref_dbm=
                        ref_dbm,

                    status=
                        "OK",
                )

                results.append(
                    result
                )

                self.result_ready.emit(
                    result
                )

                self._log(
                    f"[{idx:3d}/{n_freqs}] "
                    f"{result.freq_mhz:8.1f} MHz | "
                    f"G={result.gain_db:5.1f} dB | "
                    f"Max={result.max_dbfs:7.2f} dBFS | "
                    f"{result.max_dbm:7.2f} dBm"
                )

            except Exception as exc:

                self._log(
                    f"[{idx}/{n_freqs}] "
                    f"Erreur : {exc}"
                )

            self.progress.emit(
                int(
                    idx
                    / n_freqs
                    * 100
                )
            )

        self.scan_done.emit(
            results
        )

        self._log(
            "Scan terminé."
        )


# =============================================================
# PRPD
# =============================================================

class PrpdThread(QThread):
    """
    Acquisition et traitement PRPD sur une fréquence fixe.

    Signaux Qt :

        acq_done(
            phases,
            amps,
            info,
            spectre_freqs,
            spectre_dbfs,
            iq_samples
        )

        progress(int)
        log_message(str)
        error(str)
    """

    # ---------------------------------------------------------
    # 6 paramètres :
    #
    #   1 phases
    #   2 amplitudes
    #   3 info
    #   4 fréquences spectre
    #   5 amplitudes spectre
    #   6 IQ brut
    # ---------------------------------------------------------

    acq_done = pyqtSignal(
        object,
        object,
        dict,
        object,
        object,
        object,
    )

    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        params: dict,
        gain_calibration: dict,
    ) -> None:

        super().__init__()

        self.params = {
            **DEFAULT_PARAMS,
            **params,
        }

        self.gain_calibration = (
            gain_calibration
        )

        self._stop_flag = False

    # ---------------------------------------------------------

    def stop(self) -> None:
        self._stop_flag = True

    # ---------------------------------------------------------

    def _log(
        self,
        msg: str,
    ) -> None:

        self.log_message.emit(
            msg
        )

    # ---------------------------------------------------------
    # PRPD
    # ---------------------------------------------------------

    def run(self) -> None:

        import uhd

        p = self.params

        # =====================================================
        # Vérification paramètres communs
        # =====================================================

        nyq = (
            p["rate_hz"]
            / 2.0
        )

        requested_cutoff = float(
            p["lowpass_cutoff_hz"]
        )

        if requested_cutoff <= 0:

            self.error.emit(
                "Low-pass cutoff invalide : "
                "la valeur doit être > 0."
            )

            return

        # Même protection que pour le scan
        actual_cutoff = min(
            requested_cutoff,
            nyq * 0.95,
        )

        # =====================================================
        # CONNEXION USRP
        # =====================================================

        try:

            self._log(
                "Connexion USRP pour PRPD..."
            )

            serial_arg = (
                f"serial={p['usrp_serial']}"
                if p["usrp_serial"]
                else ""
            )

            usrp = uhd.usrp.MultiUSRP(
                serial_arg
            )

        except Exception as exc:

            self.error.emit(
                "USRP non disponible :\n"
                f"{exc}"
            )

            return

        # =====================================================
        # Informations paramètres utilisés
        # =====================================================

        self._log(
            f"PRPD : "
            f"Fs={p['rate_hz']/1e6:.3f} MS/s | "
            f"Cutoff demandé="
            f"{requested_cutoff/1e6:.3f} MHz | "
            f"Cutoff appliqué="
            f"{actual_cutoff/1e6:.3f} MHz"
        )

        # =====================================================
        # TABLEAUX POUR TOUTES LES ACQUISITIONS
        # =====================================================

        all_phases: list[np.ndarray] = []
        all_amps: list[np.ndarray] = []
        all_info: list[dict] = []

        # IQ bruts
        all_iq_samples: list[np.ndarray] = []

        n_acq = int(
            p["prpd_n_acq"]
        )

        # =====================================================
        # ACQUISITIONS PRPD
        # =====================================================

        for i in range(
            n_acq
        ):

            if self._stop_flag:
                break

            # -------------------------------------------------
            # Synchronisation 50 Hz
            # -------------------------------------------------

            self._log(
                f"Synchronisation PPS "
                f"({i + 1}/{n_acq})..."
            )

            try:

                sync_on_external_50hz_pps(
                    usrp
                )

            except Exception as exc:

                self._log(
                    f"  PPS sync échoué : "
                    f"{exc} — "
                    f"acquisition sans sync."
                )

            # -------------------------------------------------
            # Informations acquisition
            # -------------------------------------------------

            self._log(
                f"Acquisition PRPD "
                f"{i + 1}/{n_acq} — "
                f"{p['prpd_freq_hz']/1e6:.3f} MHz | "
                f"{p['prpd_gain_db']:.0f} dB | "
                f"Fs={p['rate_hz']/1e6:.1f} MS/s | "
                f"Fc={actual_cutoff/1e6:.2f} MHz | "
                f"T={p['prpd_duration_s']:.3f} s"
            )

            # -------------------------------------------------
            # Progression
            # -------------------------------------------------

            def _prog(
                pct: int,
            ) -> None:

                global_pct = int(
                    i / n_acq * 100
                    + pct / n_acq
                )

                self.progress.emit(
                    global_pct
                )

            # =================================================
            # ACQUISITION IQ
            # =================================================

            try:

                samples, t_start = (
                    rx_prpd_sync(
                        usrp,

                        freq_hz=
                            p[
                                "prpd_freq_hz"
                            ],

                        rate_hz=
                            p[
                                "rate_hz"
                            ],

                        duration_s=
                            p[
                                "prpd_duration_s"
                            ],

                        gain_db=
                            p[
                                "prpd_gain_db"
                            ],

                        channel=
                            p[
                                "channel"
                            ],

                        antenna=
                            p[
                                "antenna"
                            ],

                        progress_cb=
                            _prog,
                    )
                )

            except Exception as exc:

                self.error.emit(
                    "Erreur acquisition PRPD :\n"
                    f"{exc}"
                )

                return

            # -------------------------------------------------
            # Vérification
            # -------------------------------------------------

            if len(samples) == 0:

                self._log(
                    "  Acquisition IQ vide."
                )

                continue

            # =================================================
            # CONSERVATION IQ BRUT
            # =================================================

            iq_copy = np.asarray(
                samples,
                dtype=np.complex64,
            ).copy()

            all_iq_samples.append(
                iq_copy
            )

            self._log(
                f"  IQ capturé : "
                f"{len(iq_copy):,} samples "
                f"({iq_copy.nbytes / 1024**2:.1f} MiB)"
            )

            # =================================================
            # TRAITEMENT PRPD
            #
            # IMPORTANT :
            # On transmet ici EXACTEMENT le même
            # lowpass_cutoff_hz que celui du scan.
            # =================================================

            phases, amps, info = (
                process_pd_signal(
                    samples,

                    rate_hz=
                        p[
                            "rate_hz"
                        ],

                    t_start=
                        t_start,

                    f_offset_hz=
                        p[
                            "prpd_f_offset_hz"
                        ],

                    gain_calibration=
                        self.gain_calibration,

                    freq_hz=
                        p[
                            "prpd_freq_hz"
                        ],

                    # =========================================
                    # MÊME CUTOFF QUE LE SCAN
                    # =========================================
                    cutoff_hz=
                        p[
                            "lowpass_cutoff_hz"
                        ],
                )
            )

            # -------------------------------------------------
            # Log traitement
            # -------------------------------------------------

            self._log(
                f"  {info['n_pulses']} "
                f"pulses détectées | "
                f"Unité : "
                f"{info['unit']} | "
                f"Fc="
                f"{info.get('cutoff_hz', actual_cutoff)/1e6:.3f} MHz"
            )

            # -------------------------------------------------
            # Stockage résultats
            # -------------------------------------------------

            all_phases.append(
                phases
            )

            all_amps.append(
                amps
            )

            all_info.append(
                info
            )

        # =====================================================
        # VÉRIFICATION FIN ACQUISITION
        # =====================================================

        if not all_iq_samples:

            self.error.emit(
                "Aucune acquisition IQ valide."
            )

            return

        # =====================================================
        # CONCATÉNATION IQ
        # =====================================================

        if len(
            all_iq_samples
        ) == 1:

            iq_all = (
                all_iq_samples[0]
            )

        else:

            iq_all = np.concatenate(
                all_iq_samples
            )

        # =====================================================
        # CONCATÉNATION PRPD
        # =====================================================

        valid_phases = [
            x
            for x in all_phases
            if len(x) > 0
        ]

        valid_amps = [
            x
            for x in all_amps
            if len(x) > 0
        ]

        if valid_phases:

            phases_all = np.concatenate(
                valid_phases
            )

            amps_all = np.concatenate(
                valid_amps
            )

        else:

            phases_all = np.array(
                [],
                dtype=float,
            )

            amps_all = np.array(
                [],
                dtype=float,
            )

        # =====================================================
        # INFORMATIONS
        # =====================================================

        info_merged = {

            "n_pulses":
                int(
                    sum(
                        x.get(
                            "n_pulses",
                            0,
                        )
                        for x in all_info
                    )
                ),

            "unit":
                (
                    all_info[0].get(
                        "unit",
                        "dBFS",
                    )
                    if all_info
                    else "dBFS"
                ),

            "n_acq":
                len(
                    all_iq_samples
                ),

            "n_iq_samples":
                int(
                    len(
                        iq_all
                    )
                ),

            "iq_dtype":
                str(
                    iq_all.dtype
                ),

            "iq_size_mb":
                float(
                    iq_all.nbytes
                    / 1024**2
                ),

            "sample_rate_hz":
                float(
                    p[
                        "rate_hz"
                    ]
                ),

            "frequency_hz":
                float(
                    p[
                        "prpd_freq_hz"
                    ]
                ),

            "gain_db":
                float(
                    p[
                        "prpd_gain_db"
                    ]
                ),

            "duration_s":
                float(
                    p[
                        "prpd_duration_s"
                    ]
                ),

            # -------------------------------------------------
            # Cutoff demandé dans l'IHM
            # -------------------------------------------------

            "cutoff_requested_hz":
                float(
                    p[
                        "lowpass_cutoff_hz"
                    ]
                ),

            # -------------------------------------------------
            # Cutoff réellement utilisé
            # -------------------------------------------------

            "cutoff_hz":
                float(
                    actual_cutoff
                ),
        }

        # =====================================================
        # PAS DE FFT ICI
        # =====================================================

        # Le spectre principal est calculé par ScanThread.
        # On conserve ces tableaux vides pour compatibilité IHM.

        spec_freqs = np.array(
            [],
            dtype=float,
        )

        spec_dbfs = np.array(
            [],
            dtype=float,
        )

        # =====================================================
        # LOG FINAL
        # =====================================================

        self._log(
            f"PRPD terminé | "
            f"{len(phases_all)} pulses | "
            f"{len(iq_all):,} IQ samples | "
            f"{iq_all.nbytes / 1024**2:.1f} MiB | "
            f"Fs={p['rate_hz']/1e6:.2f} MS/s | "
            f"Fc={actual_cutoff/1e6:.2f} MHz"
        )

        # =====================================================
        # ENVOI À L'IHM
        # =====================================================

        self.acq_done.emit(
            phases_all,
            amps_all,
            info_merged,
            spec_freqs,
            spec_dbfs,
            iq_all,
        )

        self.progress.emit(
            100
        )