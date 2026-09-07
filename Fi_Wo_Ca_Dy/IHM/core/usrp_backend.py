#!/usr/bin/env python3
"""
core/usrp_backend.py
--------------------
Backend USRP pour :
  - Scan spectral avec gain fixe
  - Acquisition PRPD
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

    # Scan spectral
    "f_start_hz":        100e6,
    "f_stop_hz":         2.0e9,
    "step_hz":           12e6,

    "rate_hz":           12e6,

    # Gain FIXE pour le spectre
    "scan_gain_db":      40.0,

    "lo_offset_hz":      1e6,

    "discard_s":         0.001,
    "useful_s":          0.021,
    "settling_s":        0.100,

    # Traitement spectre
    "lowpass_cutoff_hz": 5.99e6,
    "filter_order":      4,
    "robust_percentile": 99.99,
    "filter_edge_s":     0.0002,
    "clipping_amp":      0.98,

    # PRPD
    "prpd_freq_hz":      1.196e9,
    "prpd_gain_db":      40.0,
    "prpd_duration_s":   3.0,
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

    def stop(self) -> None:
        self._stop_flag = True

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

        discard = int(
            p["discard_s"]
            * p["rate_hz"]
        )

        return samples[discard:]

    # ---------------------------------------------------------
    # Métriques
    # ---------------------------------------------------------

    def _compute_metrics(
        self,
        samples: np.ndarray,
        sos,
    ) -> dict:

        from scipy.signal import sosfiltfilt

        p = self.params

        filtered = sosfiltfilt(
            sos,
            samples,
        )

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

        env = np.abs(filtered)

        return {
            "max_dbfs":
                amplitude_to_dbfs(
                    float(np.max(env))
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
                    float(np.median(env))
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

        usrp.set_rx_rate(
            p["rate_hz"],
            p["channel"],
        )

        usrp.set_rx_antenna(
            p["antenna"],
            p["channel"],
        )

        # Gain FIXE
        usrp.set_rx_gain(
            gain_db,
            p["channel"],
        )

        tune_req = uhd.types.TuneRequest(
            freq_hz,
            p["lo_offset_hz"],
        )

        usrp.set_rx_freq(
            tune_req,
            p["channel"],
        )

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

        return actual_freq, actual_gain

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

        # Filtre
        nyq = p["rate_hz"] / 2.0

        cutoff = min(
            p["lowpass_cutoff_hz"],
            nyq * 0.95,
        )

        sos = butter(
            p["filter_order"],
            cutoff / nyq,
            btype="low",
            output="sos",
        )

        # Samples
        n_total = int(
            (
                p["discard_s"]
                + p["useful_s"]
            )
            * p["rate_hz"]
        )

        # Fréquences
        freqs_hz = np.arange(
            p["f_start_hz"],
            p["f_stop_hz"]
            + p["step_hz"] / 2,
            p["step_hz"],
        )

        n_freqs = len(freqs_hz)

        # Connexion USRP
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
            f"Gain fixe = {fixed_gain:.0f} dB"
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
                # Configuration fréquence + gain FIXE
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

                results.append(result)

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

    acq_done = pyqtSignal(
        object,
        object,
        dict,
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

    def stop(self) -> None:
        self._stop_flag = True

    def _log(self, msg: str) -> None:
        self.log_message.emit(msg)

    def run(self) -> None:

        import uhd

        p = self.params

        # Connexion USRP
        try:

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
                f"USRP non disponible :\n{exc}"
            )

            return

        all_phases = []
        all_amps = []
        all_info = []

        n_acq = p[
            "prpd_n_acq"
        ]

        # =====================================================
        # Acquisitions PRPD
        # =====================================================

        for i in range(n_acq):

            if self._stop_flag:
                break

            self._log(
                f"PRPD {i + 1}/{n_acq}"
            )

            # Synchronisation 50 Hz
            try:

                sync_on_external_50hz_pps(
                    usrp
                )

            except Exception as exc:

                self._log(
                    f"PPS non disponible : {exc}"
                )

            # Progression acquisition
            def progress_cb(pct):

                self.progress.emit(
                    int(
                        i / n_acq * 100
                        + pct / n_acq
                    )
                )

            # Acquisition
            try:

                samples, t_start = (
                    rx_prpd_sync(
                        usrp,

                        freq_hz=
                            p["prpd_freq_hz"],

                        rate_hz=
                            p["rate_hz"],

                        duration_s=
                            p["prpd_duration_s"],

                        gain_db=
                            p["prpd_gain_db"],

                        channel=
                            p["channel"],

                        antenna=
                            p["antenna"],

                        progress_cb=
                            progress_cb,
                    )
                )

            except Exception as exc:

                self.error.emit(
                    f"Erreur PRPD :\n{exc}"
                )

                return

            if len(samples) == 0:
                continue

            # Traitement
            phases, amps, info = (
                process_pd_signal(
                    samples,

                    rate_hz=
                        p["rate_hz"],

                    t_start=
                        t_start,

                    f_offset_hz=
                        p["prpd_f_offset_hz"],

                    gain_calibration=
                        self.gain_calibration,

                    freq_hz=
                        p["prpd_freq_hz"],
                )
            )

            all_phases.append(
                phases
            )

            all_amps.append(
                amps
            )

            all_info.append(
                info
            )

        if not all_info:

            self.error.emit(
                "Aucune acquisition PRPD valide."
            )

            return

        # =====================================================
        # Fusion
        # =====================================================

        valid_phases = [
            x
            for x in all_phases
            if len(x)
        ]

        valid_amps = [
            x
            for x in all_amps
            if len(x)
        ]

        phases_all = (
            np.concatenate(valid_phases)
            if valid_phases
            else np.array([])
        )

        amps_all = (
            np.concatenate(valid_amps)
            if valid_amps
            else np.array([])
        )

        info = {
            "n_pulses":
                sum(
                    x["n_pulses"]
                    for x in all_info
                ),

            "unit":
                all_info[0]["unit"],

            "n_acq":
                len(all_info),
        }

        # Pas de spectre FFT PRPD
        spec_freqs = np.array([])
        spec_dbfs = np.array([])

        self.acq_done.emit(
            phases_all,
            amps_all,
            info,
            spec_freqs,
            spec_dbfs,
        )

        self.progress.emit(100)

        self._log(
            "PRPD terminé."
        )