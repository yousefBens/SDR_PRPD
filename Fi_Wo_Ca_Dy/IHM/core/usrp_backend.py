#!/usr/bin/env python3
"""
core/usrp_backend.py
--------------------
QThread gérant toutes les opérations USRP de manière non-bloquante.

Deux modes :
  - ScanThread  : scan spectral 100 MHz → 2 GHz avec gain auto
  - PrpdThread  : acquisition PRPD sur une fréquence fixe
  - DemoThread  : simulation sans USRP (pour test IHM)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.calibration import (
    amplitude_to_dbfs,
    build_all_gain_calibrations,
    dbfs_to_dbm,
    load_power_calibration,
)
from core.auto_gain import (
    GAIN_START_DB,
    GAINS_DB,
    decide_next_gain,
)
from core.prpd_processor import (
    process_pd_signal,
    rx_prpd_sync,
    sync_on_external_50hz_pps,
)

# ──────────────────────────────────────────────────────────────
# Dataclass résultat fréquence
# ──────────────────────────────────────────────────────────────

@dataclass
class FreqResult:
    freq_mhz: float
    freq_hz:  float
    gain_db:  float
    max_dbfs:  float
    robust_dbfs: float
    median_dbfs: float
    clipping_fraction: float
    max_dbm:    float
    robust_dbm: float
    ref_dbm:    float
    status:     str       # "OK", "SAT@MIN_GAIN", "WEAK@MAX_GAIN", etc.


# ──────────────────────────────────────────────────────────────
# Paramètres USRP par défaut (modifiables depuis l'UI)
# ──────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "usrp_serial":        "306BD15",
    "channel":            0,
    "antenna":            "RX2",
    "f_start_hz":         100e6,
    "f_stop_hz":          2.0e9,
    "rate_hz":            12e6,
    "step_hz":            12e6,
    "lo_offset_hz":       1e6,
    "discard_s":          0.001,
    "useful_s":           0.021,
    "settling_s":         0.10,
    "repeats":            1,
    "lowpass_cutoff_hz":  5.99e6,
    "filter_order":       4,
    "robust_percentile":  99.99,
    "filter_edge_s":      0.0002,
    "clipping_amp":       0.98,
    "prpd_freq_hz":       1.196e9,
    "prpd_gain_db":       40.0,
    "prpd_duration_s":    3.0,
    "prpd_f_offset_hz":   0.0,
    "prpd_n_acq":         1,
    "demo_mode":          False,
}


# ──────────────────────────────────────────────────────────────
# Thread scan spectral
# ──────────────────────────────────────────────────────────────

class ScanThread(QThread):
    """
    Scan 100 MHz → 2 GHz avec gestion automatique du gain.

    Signaux Qt :
      result_ready(FreqResult)    : émis après chaque fréquence
      progress(int)               : 0 → 100 %
      scan_done(list[FreqResult]) : fin du scan
      log_message(str)            : message de log
      error(str)                  : erreur fatale
    """

    result_ready = pyqtSignal(object)   # FreqResult
    progress     = pyqtSignal(int)
    scan_done    = pyqtSignal(list)
    log_message  = pyqtSignal(str)
    error        = pyqtSignal(str)

    def __init__(self, params: dict, cal_calibrations: dict) -> None:
        super().__init__()
        self.params = {**DEFAULT_PARAMS, **params}
        self.cal_calibrations = cal_calibrations  # {gain_db: cal_dict}
        self._stop_flag = False

    def stop(self) -> None:
        self._stop_flag = True

    # ── Helpers signal ────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self.log_message.emit(msg)

    # ── Acquisition d'un buffer ───────────────────────────────

    def _acquire(self, usrp, n_samples: int) -> np.ndarray:
        import uhd

        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        stream_args.channels = [self.params["channel"]]
        rx_streamer = usrp.get_rx_stream(stream_args)
        rx_md = uhd.types.RXMetadata()

        samples = np.zeros(n_samples, dtype=np.complex64)
        buff = np.zeros(
            (1, rx_streamer.get_max_num_samps()), dtype=np.complex64
        )

        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        cmd.num_samps  = n_samples
        cmd.stream_now = True
        rx_streamer.issue_stream_cmd(cmd)

        total = 0
        while total < n_samples:
            n = rx_streamer.recv(buff, rx_md, timeout=3.0)
            if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
                raise RuntimeError(f"UHD RX error: {rx_md.strerror()}")
            if n <= 0:
                raise RuntimeError("Aucun échantillon reçu.")
            count = min(n, n_samples - total)
            samples[total: total + count] = buff[0, :count]
            total += count

        discard = int(self.params["discard_s"] * self.params["rate_hz"])
        return samples[discard:]

    # ── Calcul métriques ────────────────────────────────────

    def _compute_metrics(
        self,
        samples: np.ndarray,
        sos,
    ) -> dict:
        from scipy.signal import sosfiltfilt

        p = self.params
        filtered = sosfiltfilt(sos, samples)

        edge = int(p["filter_edge_s"] * p["rate_hz"])
        if edge > 0 and filtered.size > 2 * edge:
            filtered = filtered[edge:-edge]

        env = np.abs(filtered)
        clip_amp = p["clipping_amp"]

        return {
            "max_dbfs":   amplitude_to_dbfs(float(np.max(env))),
            "robust_dbfs": amplitude_to_dbfs(
                float(np.percentile(env, p["robust_percentile"]))
            ),
            "median_dbfs": amplitude_to_dbfs(float(np.median(env))),
            "clipping_fraction": float(
                np.mean(np.abs(samples) >= clip_amp)
            ),
        }

    # ── Configuration fréquence + gain ───────────────────────

    def _tune(self, usrp, freq_hz: float, gain_db: float) -> tuple[float, float]:
        import uhd

        p = self.params
        usrp.set_rx_rate(p["rate_hz"], p["channel"])
        usrp.set_rx_antenna(p["antenna"], p["channel"])
        usrp.set_rx_gain(gain_db, p["channel"])

        tune_req = uhd.types.TuneRequest(freq_hz, p["lo_offset_hz"])
        usrp.set_rx_freq(tune_req, p["channel"])
        usrp.set_rx_dc_offset(True, p["channel"])
        usrp.set_rx_iq_balance(True, p["channel"])

        time.sleep(p["settling_s"])

        actual_freq = float(usrp.get_rx_freq(p["channel"]))
        actual_gain = float(usrp.get_rx_gain(p["channel"]))
        return actual_freq, actual_gain

    # ── Run ───────────────────────────────────────────────────

    def run(self) -> None:
        import uhd
        from scipy.signal import butter

        p = self.params

        # =====================================================
        # Filtre passe-bas
        # =====================================================

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
            p["f_stop_hz"] + p["step_hz"] / 2.0,
            p["step_hz"],
            dtype=float,
        )

        n_freqs = len(
            freqs_hz
        )

        # =====================================================
        # Connexion USRP
        # =====================================================

        try:
            self._log(
                "Connexion USRP..."
            )

            serial_arg = (
                f"serial={p['usrp_serial']}"
                if p["usrp_serial"]
                else ""
            )

            usrp = uhd.usrp.MultiUSRP(
                serial_arg
            )

            self._log(
                f"USRP connecté — "
                f"{n_freqs} fréquences à scanner."
            )

        except Exception as exc:

            self.error.emit(
                "Impossible de connecter l'USRP :\n"
                f"{exc}"
            )

            return

        results: list[FreqResult] = []

        # =====================================================
        # Fonction locale :
        # calibration correspondant au gain réellement utilisé
        # =====================================================

        def get_calibration_for_gain(
            gain_db: float,
        ) -> tuple[float, dict]:

            if not self.cal_calibrations:
                raise RuntimeError(
                    "Aucune calibration disponible."
                )

            available_gains = sorted(
                float(g)
                for g in self.cal_calibrations.keys()
            )

            available_array = np.asarray(
                available_gains,
                dtype=float,
            )

            selected_gain = float(
                available_array[
                    np.argmin(
                        np.abs(
                            available_array
                            - float(gain_db)
                        )
                    )
                ]
            )

            return (
                selected_gain,
                self.cal_calibrations[
                    selected_gain
                ],
            )

        # =====================================================
        # SCAN
        # =====================================================

        for idx, req_freq_hz in enumerate(
            freqs_hz,
            start=1,
        ):

            if self._stop_flag:

                self._log(
                    "Scan interrompu par l'utilisateur."
                )

                break

            # -------------------------------------------------
            # À CHAQUE fréquence on repart du gain initial
            # -------------------------------------------------

            current_gain = float(
                GAIN_START_DB
            )

            status = "OK"

            # Variables finales de la fréquence courante
            actual_freq = float(
                req_freq_hz
            )

            actual_gain = float(
                current_gain
            )

            m = None

            max_dbm = float("nan")
            robust_dbm = float("nan")
            ref_dbm = float("nan")

            # -------------------------------------------------
            # Maximum 2 acquisitions par fréquence :
            #
            # attempt = 0 :
            #     acquisition au gain initial
            #
            # attempt = 1 :
            #     acquisition avec le gain corrigé, si nécessaire
            # -------------------------------------------------

            for attempt in range(2):

                if self._stop_flag:
                    break

                # =============================================
                # 1) Accord fréquence + gain
                # =============================================

                try:

                    (
                        actual_freq,
                        actual_gain,
                    ) = self._tune(
                        usrp,
                        float(
                            req_freq_hz
                        ),
                        float(
                            current_gain
                        ),
                    )

                    # =========================================
                    # 2) Acquisition
                    # =========================================

                    samples = self._acquire(
                        usrp,
                        n_total,
                    )

                    # =========================================
                    # 3) Métriques temporelles
                    # =========================================

                    m = self._compute_metrics(
                        samples,
                        sos,
                    )

                except Exception as exc:

                    self._log(
                        f"  [{idx}/{n_freqs}] "
                        f"Erreur acquisition : {exc}"
                    )

                    status = "ERROR"

                    break

                # =============================================
                # 4) Calibration du gain réellement utilisé
                # =============================================

                try:

                    (
                        calibrated_gain,
                        cal,
                    ) = get_calibration_for_gain(
                        actual_gain
                    )

                    max_dbm, ref_dbm = dbfs_to_dbm(
                        cal,
                        m["max_dbfs"],
                        actual_freq,
                    )

                    robust_dbm, _ = dbfs_to_dbm(
                        cal,
                        m["robust_dbfs"],
                        actual_freq,
                    )

                except Exception as exc:

                    self._log(
                        f"  [{idx}/{n_freqs}] "
                        f"Erreur calibration : {exc}"
                    )

                    status = "CAL_ERROR"

                    max_dbm = float("nan")
                    robust_dbm = float("nan")
                    ref_dbm = float("nan")

                    break

                # =============================================
                # 5) Décision basée sur Pmin/Pmax en dBm
                # =============================================

                try:

                    decision = decide_next_gain(
                        frequency_hz=
                            actual_freq,

                        current_gain_db=
                            actual_gain,

                        measured_dbm=
                            max_dbm,

                        clipping_fraction=
                            m[
                                "clipping_fraction"
                            ],
                    )

                except Exception as exc:

                    self._log(
                        f"  [{idx}/{n_freqs}] "
                        f"Erreur auto-gain : {exc}"
                    )

                    status = "AUTO_GAIN_ERROR"

                    break

                # =============================================
                # Log détaillé de la décision
                # =============================================

                self._log(
                    f"  [{idx}/{n_freqs}] "
                    f"{actual_freq / 1e6:.1f} MHz | "
                    f"Acq {attempt + 1}/2 | "
                    f"G={actual_gain:.0f} dB | "
                    f"Max={m['max_dbfs']:.2f} dBFS | "
                    f"P={max_dbm:.2f} dBm | "
                    f"Pmin={decision.pmin_dbm:.2f} dBm | "
                    f"Pmax={decision.pmax_dbm:.2f} dBm | "
                    f"Zone=["
                    f"{decision.lower_limit_dbm:.2f}, "
                    f"{decision.upper_limit_dbm:.2f}"
                    f"] dBm | "
                    f"{decision.reason}"
                )

                # =============================================
                # 6) Première acquisition
                # =============================================

                if attempt == 0:

                    # Gain correct dès la première acquisition
                    if not decision.needs_second_acquisition:

                        status = decision.reason

                        break

                    # Gain à corriger :
                    # deuxième acquisition à LA MÊME fréquence
                    if decision.next_gain_db is not None:

                        current_gain = float(
                            decision.next_gain_db
                        )

                        status = decision.reason

                        self._log(
                            f"      → Nouvelle acquisition "
                            f"à {actual_freq / 1e6:.1f} MHz "
                            f"avec gain "
                            f"{current_gain:.0f} dB"
                        )

                        continue

                    # Aucun autre gain possible
                    status = decision.reason

                    break

                # =============================================
                # 7) Deuxième acquisition
                # =============================================

                else:

                    # On conserve obligatoirement cette deuxième
                    # acquisition et on passe à la fréquence suivante.
                    #
                    # Même si la deuxième mesure demanderait encore
                    # un autre gain, on ne fait PAS de troisième
                    # acquisition.
                    if decision.needs_second_acquisition:

                        status = (
                            "SECOND_ACQ_LIMIT | "
                            f"{decision.reason}"
                        )

                    else:

                        status = (
                            "SECOND_ACQ_OK | "
                            f"{decision.reason}"
                        )

                    break

            # -------------------------------------------------
            # Si erreur avant d'obtenir une mesure valide
            # -------------------------------------------------

            if m is None:

                self._log(
                    f"[{idx:3d}/{n_freqs}] "
                    f"{req_freq_hz / 1e6:8.1f} MHz | "
                    f"Aucune mesure valide."
                )

                self.progress.emit(
                    int(
                        idx
                        / n_freqs
                        * 100
                    )
                )

                continue

            # =================================================
            # Résultat FINAL de cette fréquence
            #
            # Il correspond :
            #   - à la 1re acquisition si le gain initial était OK
            #   - à la 2e acquisition si le gain a été modifié
            # =================================================

            result = FreqResult(

                freq_mhz=
                    actual_freq
                    / 1e6,

                freq_hz=
                    actual_freq,

                gain_db=
                    actual_gain,

                max_dbfs=
                    m[
                        "max_dbfs"
                    ],

                robust_dbfs=
                    m[
                        "robust_dbfs"
                    ],

                median_dbfs=
                    m[
                        "median_dbfs"
                    ],

                clipping_fraction=
                    m[
                        "clipping_fraction"
                    ],

                max_dbm=
                    max_dbm,

                robust_dbm=
                    robust_dbm,

                ref_dbm=
                    ref_dbm,

                status=
                    status,
            )

            results.append(
                result
            )

            self.result_ready.emit(
                result
            )

            self.progress.emit(
                int(
                    idx
                    / n_freqs
                    * 100
                )
            )

            self._log(
                f"[{idx:3d}/{n_freqs}] "
                f"{result.freq_mhz:8.1f} MHz | "
                f"Gfinal={result.gain_db:5.1f} dB | "
                f"Max={result.max_dbfs:7.2f} dBFS | "
                f"Max≈{result.max_dbm:7.2f} dBm | "
                f"Clip={result.clipping_fraction:.2e} | "
                f"{result.status}"
            )

        # =====================================================
        # Fin du scan
        # =====================================================

        self.scan_done.emit(
            results
        )

        self._log(
            "Scan terminé."
        )


# ──────────────────────────────────────────────────────────────
# Thread PRPD
# ──────────────────────────────────────────────────────────────

class PrpdThread(QThread):
    """
    Acquisition et traitement PRPD sur une fréquence fixe.

    Signaux Qt :
      acq_done(phases, amps, info, spectre_freqs, spectre_dbfs)
      progress(int)
      log_message(str)
      error(str)
    """

    acq_done    = pyqtSignal(object, object, dict, object, object)
    progress    = pyqtSignal(int)
    log_message = pyqtSignal(str)
    error       = pyqtSignal(str)

    def __init__(
        self,
        params: dict,
        gain_calibration: dict,
    ) -> None:
        super().__init__()
        self.params = {**DEFAULT_PARAMS, **params}
        self.gain_calibration = gain_calibration
        self._stop_flag = False

    def stop(self) -> None:
        self._stop_flag = True

    def _log(self, msg: str) -> None:
        self.log_message.emit(msg)

    def run(self) -> None:
        import uhd

        p = self.params

        try:
            self._log("Connexion USRP pour PRPD...")
            serial_arg = f"serial={p['usrp_serial']}" if p["usrp_serial"] else ""
            usrp = uhd.usrp.MultiUSRP(serial_arg)
        except Exception as exc:
            self.error.emit(f"USRP non disponible :\n{exc}")
            return

        all_phases: list[np.ndarray] = []
        all_amps:   list[np.ndarray] = []
        all_info:   list[dict]       = []

        n_acq = p["prpd_n_acq"]

        for i in range(n_acq):
            if self._stop_flag:
                break

            self._log(f"Synchronisation PPS ({i+1}/{n_acq})...")
            try:
                sync_on_external_50hz_pps(usrp)
            except Exception as exc:
                self._log(f"  PPS sync échoué : {exc} — acquisition sans sync.")

            self._log(
                f"Acquisition PRPD {i+1}/{n_acq} — "
                f"{p['prpd_freq_hz']/1e6:.3f} MHz | "
                f"{p['prpd_gain_db']:.0f} dB"
            )

            def _prog(pct: int) -> None:
                self.progress.emit(int(i / n_acq * 100 + pct / n_acq))

            try:
                samples, t_start = rx_prpd_sync(
                    usrp,
                    freq_hz=p["prpd_freq_hz"],
                    rate_hz=p["rate_hz"],
                    duration_s=p["prpd_duration_s"],
                    gain_db=p["prpd_gain_db"],
                    channel=p["channel"],
                    antenna=p["antenna"],
                    progress_cb=_prog,
                )
            except Exception as exc:
                self.error.emit(f"Erreur acquisition PRPD :\n{exc}")
                return

            if len(samples) == 0:
                self._log("  Acquisition vide.")
                continue

            phases, amps, info = process_pd_signal(
                samples,
                rate_hz=p["rate_hz"],
                t_start=t_start,
                f_offset_hz=p["prpd_f_offset_hz"],
                gain_calibration=self.gain_calibration,
                freq_hz=p["prpd_freq_hz"],
            )

            self._log(
                f"  {info['n_pulses']} pulses détectées | "
                f"Unité : {info['unit']}"
            )

            all_phases.append(phases)
            all_amps.append(amps)
            all_info.append(info)

        if not all_phases:
            self.error.emit("Aucune pulse détectée.")
            return

        # Concaténation de toutes les acquisitions
        phases_all = np.concatenate([p for p in all_phases if len(p)])
        amps_all   = np.concatenate([a for a in all_amps   if len(a)])
        info_merged = {
            "n_pulses": sum(i["n_pulses"] for i in all_info),
            "unit":     all_info[0]["unit"] if all_info else "dBFS",
            "n_acq":    n_acq,
        }

        # Pas de calcul FFT / spectre : on conserve deux tableaux vides
        # pour garder la compatibilité avec le signal Qt acq_done existant.
        spec_freqs = np.array([])
        spec_dbfs  = np.array([])

        self.acq_done.emit(phases_all, amps_all, info_merged, spec_freqs, spec_dbfs)
        self.progress.emit(100)
        self._log("PRPD terminé.")


# ──────────────────────────────────────────────────────────────
# Thread de démonstration (sans USRP)
# ──────────────────────────────────────────────────────────────

class DemoScanThread(QThread):
    """
    Simule un scan spectral pour tester l'IHM sans USRP.
    Génère des données réalistes avec quelques « pics » de décharge.
    """

    result_ready = pyqtSignal(object)
    progress     = pyqtSignal(int)
    scan_done    = pyqtSignal(list)
    log_message  = pyqtSignal(str)
    error        = pyqtSignal(str)

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = {**DEFAULT_PARAMS, **params}
        self._stop_flag = False

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        p = self.params
        rng = np.random.default_rng(42)

        freqs_hz = np.arange(
            p["f_start_hz"],
            p["f_stop_hz"] + p["step_hz"] / 2.0,
            p["step_hz"],
            dtype=float,
        )
        n_freqs = len(freqs_hz)

        # Bruit de fond simulé
        noise_floor_dbm = -90.0

        # Pics simulés de décharge (typiques PD)
        pd_centers = [450e6, 900e6, 1250e6, 1700e6]
        pd_power   = [-55.0, -62.0, -48.0, -70.0]
        pd_bw      = [60e6, 80e6, 40e6, 50e6]

        results: list[FreqResult] = []
        self.log_message.emit(f"[DÉMO] Scan simulé — {n_freqs} fréquences.")

        for idx, freq_hz in enumerate(freqs_hz, start=1):
            if self._stop_flag:
                break

            time.sleep(0.02)  # Simulation du temps d'acquisition

            # Puissance simulée
            dbm = noise_floor_dbm + rng.normal(0, 1.5)
            for fc, pw, bw in zip(pd_centers, pd_power, pd_bw):
                dbm = max(
                    dbm,
                    pw - 10.0 * ((freq_hz - fc) / bw) ** 2
                    + rng.normal(0, 2.0),
                )

            # Gain auto simulé
            if dbm > -50:
                gain = 40.0
            elif dbm > -65:
                gain = 60.0
            else:
                gain = 76.0

            max_dbfs   = dbm - (-40.0 + (gain - 40.0) * 0.5)
            robust_dbfs = max_dbfs - 3.0
            median_dbfs = max_dbfs - 20.0
            clip_frac   = max(0.0, (max_dbfs + 2.0) / 100.0)

            result = FreqResult(
                freq_mhz=freq_hz / 1e6,
                freq_hz=freq_hz,
                gain_db=gain,
                max_dbfs=max_dbfs,
                robust_dbfs=robust_dbfs,
                median_dbfs=median_dbfs,
                clipping_fraction=clip_frac,
                max_dbm=dbm,
                robust_dbm=dbm - 3.0,
                ref_dbm=-40.0 + (gain - 40.0) * 0.5,
                status="DEMO",
            )

            results.append(result)
            self.result_ready.emit(result)
            self.progress.emit(int(idx / n_freqs * 100))

            if idx % 10 == 0:
                self.log_message.emit(
                    f"[DÉMO] {freq_hz/1e6:.0f} MHz | "
                    f"G={gain:.0f} dB | {dbm:.1f} dBm"
                )

        self.scan_done.emit(results)
        self.log_message.emit("[DÉMO] Scan terminé.")


class DemoPrpdThread(QThread):
    """Simule une acquisition PRPD sans USRP."""

    acq_done    = pyqtSignal(object, object, dict, object, object)
    progress    = pyqtSignal(int)
    log_message = pyqtSignal(str)
    error       = pyqtSignal(str)

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = {**DEFAULT_PARAMS, **params}
        self._stop_flag = False

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        p = self.params
        rng = np.random.default_rng(123)

        self.log_message.emit("[DÉMO] Simulation PRPD...")

        for pct in range(0, 101, 5):
            if self._stop_flag:
                return
            time.sleep(0.05)
            self.progress.emit(pct)

        # Simulation de pulses PD typiques
        n_pulses = 400
        # Concentrées autour de 90° et 270° (décharge typique AC)
        phases_90  = rng.normal(90,  15, n_pulses // 2) % 360
        phases_270 = rng.normal(270, 15, n_pulses // 2) % 360
        phases = np.concatenate([phases_90, phases_270])

        # Amplitudes simulées
        base_dbm = -60.0 if p["demo_mode"] else (
            p["prpd_gain_db"] * -0.5 - 30.0
        )
        amps = rng.normal(base_dbm, 5.0, n_pulses)
        amps += rng.exponential(3.0, n_pulses) * rng.choice([-1, 1], n_pulses)

        # Pas de spectre simulé : compatibilité avec acq_done
        spec_freqs = np.array([])
        spec_dbfs  = np.array([])

        info = {
            "n_pulses": n_pulses,
            "unit":     "dBm (simulé)",
            "n_acq":    1,
        }

        self.acq_done.emit(phases, amps, info, spec_freqs, spec_dbfs)
        self.progress.emit(100)
        self.log_message.emit(f"[DÉMO] {n_pulses} pulses simulées.")