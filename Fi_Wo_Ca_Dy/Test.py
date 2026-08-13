#!/usr/bin/env python3
from __future__ import annotations

import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvisa
import uhd

from scipy.signal import butter, sosfiltfilt


# ============================================================
# CONFIGURATION
# ============================================================

USRP_SERIAL = "306BD15"
CHANNEL = 0
ANTENNA = "RX2"

# ------------------------------------------------------------
# Fréquence et gain à tester
# ------------------------------------------------------------

FREQUENCY_HZ = 1.0e9
GAIN_DB = 60.0


# ------------------------------------------------------------
# USRP
# ------------------------------------------------------------

RATE_HZ = 12e6

DISCARD_DURATION_S = 0.002
USEFUL_DURATION_S = 0.020

ACQUISITION_DURATION_S = (
    DISCARD_DURATION_S
    + USEFUL_DURATION_S
)

SETTLING_TIME_S = 0.15

REPEATS_PER_POWER = 5


# ============================================================
# TRAITEMENT MAX
# IDENTIQUE AU CODE DU SPECTRE
# ============================================================

LOWPASS_CUTOFF_HZ = 4e6
FILTER_ORDER = 4

# On supprime les transitoires du filtre
FILTER_EDGE_DURATION_S = 0.0002

EPSILON = 1e-12


# ============================================================
# GENERATEUR N5183A
# ============================================================

GENERATOR_IP = "192.168.1.100"

GENERATOR_MIN_DBM = -20.0
GENERATOR_MAX_DBM = +20.0

GENERATOR_STEP_DB = 1.0


# ============================================================
# ATTENUATION
# ============================================================

ATTENUATION_DB = 60.0

RX2_SAFE_MAX_DBM = -20.0


# ============================================================
# CALIBRATION
# ============================================================

CAL_FILE = Path(
    "/home/yousef/Documents/testing_scripts/GEVernova/"
    "Fi_Wo_Ca_Dy/Calib/Cal_Files/"
    "rx2_power_calibration_final.pickle"
)

GAIN_TOLERANCE_DB = 0.5

FREQUENCY_TOLERANCE_HZ = 1e3


# ============================================================
# N5183A
# ============================================================

class N5183A:

    def __init__(
        self,
        ip: str,
    ):

        self.rm = pyvisa.ResourceManager(
            "@py"
        )

        resources = [

            f"TCPIP0::{ip}::inst0::INSTR",

            f"TCPIP0::{ip}::5025::SOCKET",
        ]

        self.dev = None
        self.resource = None

        last_error = None

        for resource in resources:

            print(
                f"[N5183A] Essai : {resource}"
            )

            dev = None

            try:

                dev = self.rm.open_resource(
                    resource,
                    open_timeout=5000,
                )

                dev.timeout = 10000

                dev.write_termination = "\n"
                dev.read_termination = "\n"

                identity = dev.query(
                    "*IDN?"
                ).strip()

                print(
                    f"[N5183A] {identity}"
                )

                if (
                    "N5183A"
                    not in identity.upper()
                ):

                    raise RuntimeError(
                        f"Instrument inattendu : "
                        f"{identity}"
                    )

                self.dev = dev
                self.resource = resource

                break

            except Exception as error:

                last_error = error

                if dev is not None:

                    try:
                        dev.close()

                    except Exception:
                        pass

        if self.dev is None:

            try:
                self.rm.close()

            except Exception:
                pass

            raise RuntimeError(
                "Impossible de connecter "
                f"le N5183A : {last_error}"
            )

        print(
            f"[N5183A] Ressource : "
            f"{self.resource}"
        )

        self.dev.write(
            ":OUTP OFF"
        )

        self.dev.write(
            ":FREQ:MODE FIX"
        )

        self.dev.write(
            ":POW:MODE FIX"
        )

        self._check_errors(
            "initialisation"
        )


    def output(
        self,
        enabled: bool,
    ):

        self.dev.write(
            ":OUTP ON"
            if enabled
            else ":OUTP OFF"
        )

        state = int(
            float(
                self.dev.query(
                    ":OUTP?"
                )
            )
        )

        if state != int(enabled):

            raise RuntimeError(
                "Échec changement état RF"
            )

        self._check_errors(
            "RF output"
        )


    def set_frequency(
        self,
        frequency_hz: float,
    ) -> float:

        frequency_hz = float(
            frequency_hz
        )

        self.dev.write(
            f":FREQ "
            f"{frequency_hz:.3f} HZ"
        )

        actual = float(
            self.dev.query(
                ":FREQ?"
            )
        )

        self._check_errors(
            "fréquence"
        )

        return actual


    def set_power(
        self,
        power_dbm: float,
    ) -> float:

        power_dbm = float(
            power_dbm
        )

        if not (
            GENERATOR_MIN_DBM
            <= power_dbm
            <= GENERATOR_MAX_DBM
        ):

            raise RuntimeError(
                "Puissance N5183A interdite : "
                f"{power_dbm:.2f} dBm"
            )

        expected_rx2_dbm = (
            power_dbm
            - ATTENUATION_DB
        )

        if (
            expected_rx2_dbm
            > RX2_SAFE_MAX_DBM
        ):

            raise RuntimeError(
                "ARRÊT SÉCURITÉ : "
                f"RX2 recevrait "
                f"{expected_rx2_dbm:.2f} dBm"
            )

        self.dev.write(
            f":POW "
            f"{power_dbm:.3f} DBM"
        )

        actual = float(
            self.dev.query(
                ":POW?"
            )
        )

        self._check_errors(
            "puissance"
        )

        if (
            abs(
                actual
                - power_dbm
            )
            > 0.10
        ):

            self.output(
                False
            )

            raise RuntimeError(
                f"Pgen demandée="
                f"{power_dbm:.3f} dBm, "
                f"réelle="
                f"{actual:.3f} dBm"
            )

        return actual


    def _check_errors(
        self,
        operation: str,
    ):

        errors = []

        while True:

            response = self.dev.query(
                "SYST:ERR?"
            ).strip()

            if (
                response.startswith("+0")
                or
                response.startswith("0")
            ):

                break

            errors.append(
                response
            )

        if errors:

            raise RuntimeError(
                f"Erreur SCPI "
                f"({operation}) : "
                + " | ".join(errors)
            )


    def close(
        self,
    ):

        try:

            if self.dev is not None:

                try:

                    self.dev.write(
                        ":OUTP OFF"
                    )

                except Exception:
                    pass

        finally:

            try:

                if self.dev is not None:

                    self.dev.close()

            finally:

                try:

                    self.rm.close()

                except Exception:
                    pass


# ============================================================
# CHARGEMENT CALIBRATION
# ============================================================

def load_calibration() -> dict:

    if not CAL_FILE.exists():

        raise FileNotFoundError(
            "Calibration introuvable : "
            f"{CAL_FILE}"
        )

    with CAL_FILE.open(
        "rb"
    ) as file:

        calibration = pickle.load(
            file
        )

    if CHANNEL not in calibration:

        raise ValueError(
            f"Canal {CHANNEL} "
            "absent de la calibration"
        )

    if (
        ANTENNA
        not in calibration[CHANNEL]
    ):

        raise ValueError(
            f"Antenne {ANTENNA} "
            "absente de la calibration"
        )

    raw_table = (
        calibration[
            CHANNEL
        ][
            ANTENNA
        ]
    )

    table = {}

    for (
        frequency_hz,
        gain_table,
    ) in raw_table.items():

        table[
            float(
                frequency_hz
            )
        ] = {

            float(gain_db):
                float(reference_dbm)

            for (
                gain_db,
                reference_dbm,
            )
            in gain_table.items()
        }

    return table


# ============================================================
# SELECTION DU GAIN
# ============================================================

def select_gain(
    calibration_table: dict,
    requested_gain_db: float,
) -> float:

    gains = sorted(
        {
            float(gain_db)

            for gain_table
            in calibration_table.values()

            for gain_db
            in gain_table.keys()
        }
    )

    if not gains:

        raise ValueError(
            "Aucun gain dans "
            "la calibration"
        )

    gains = np.asarray(
        gains,
        dtype=float,
    )

    selected_gain = float(
        gains[
            np.argmin(
                np.abs(
                    gains
                    - requested_gain_db
                )
            )
        ]
    )

    if (
        abs(
            selected_gain
            - requested_gain_db
        )
        > GAIN_TOLERANCE_DB
    ):

        raise ValueError(
            f"Gain demandé="
            f"{requested_gain_db:.1f} dB, "
            f"gain calibré le plus proche="
            f"{selected_gain:.1f} dB"
        )

    return selected_gain


# ============================================================
# EXTRACTION CALIBRATION POUR UN GAIN
# ============================================================

def build_gain_calibration(
    calibration_table: dict,
    gain_db: float,
):

    selected_gain = select_gain(
        calibration_table,
        gain_db,
    )

    frequencies = []
    references = []

    for frequency_hz in sorted(
        calibration_table.keys()
    ):

        gain_table = (
            calibration_table[
                frequency_hz
            ]
        )

        if (
            selected_gain
            not in gain_table
        ):

            continue

        frequencies.append(
            float(
                frequency_hz
            )
        )

        references.append(
            float(
                gain_table[
                    selected_gain
                ]
            )
        )

    if not frequencies:

        raise RuntimeError(
            "Aucune calibration "
            f"pour gain "
            f"{selected_gain:.1f} dB"
        )

    return (

        selected_gain,

        np.asarray(
            frequencies,
            dtype=float,
        ),

        np.asarray(
            references,
            dtype=float,
        ),
    )


# ============================================================
# INTERPOLATION DE LA REFERENCE
# ============================================================

def get_reference_dbm(
    frequencies_hz: np.ndarray,
    references_dbm: np.ndarray,
    frequency_hz: float,
) -> float:

    frequency_hz = float(
        frequency_hz
    )

    fmin = float(
        frequencies_hz.min()
    )

    fmax = float(
        frequencies_hz.max()
    )

    if (
        frequency_hz
        < fmin
    ):

        if (
            fmin
            - frequency_hz
            <= FREQUENCY_TOLERANCE_HZ
        ):

            frequency_hz = fmin

        else:

            raise ValueError(
                f"{frequency_hz/1e6:.3f} MHz "
                "sous la plage calibrée"
            )

    if (
        frequency_hz
        > fmax
    ):

        if (
            frequency_hz
            - fmax
            <= FREQUENCY_TOLERANCE_HZ
        ):

            frequency_hz = fmax

        else:

            raise ValueError(
                f"{frequency_hz/1e6:.3f} MHz "
                "au-dessus de la plage calibrée"
            )

    return float(
        np.interp(
            frequency_hz,
            frequencies_hz,
            references_dbm,
        )
    )


# ============================================================
# CREATION DU FILTRE PASSE-BAS
# ============================================================

def create_lowpass_filter(
    rate_hz: float,
) -> np.ndarray:

    nyquist_hz = (
        rate_hz
        / 2.0
    )

    if not (
        0.0
        < LOWPASS_CUTOFF_HZ
        < nyquist_hz
    ):

        raise ValueError(
            "LOWPASS_CUTOFF_HZ "
            "doit être entre 0 et "
            f"{nyquist_hz/1e6:.2f} MHz"
        )

    sos = butter(

        FILTER_ORDER,

        LOWPASS_CUTOFF_HZ
        / nyquist_hz,

        btype="low",

        output="sos",
    )

    return sos


# ============================================================
# CONFIGURATION USRP
# ============================================================

def configure_usrp(
    usrp: uhd.usrp.MultiUSRP,
):

    usrp.set_rx_rate(
        RATE_HZ,
        CHANNEL,
    )

    usrp.set_rx_antenna(
        ANTENNA,
        CHANNEL,
    )

    usrp.set_rx_gain(
        GAIN_DB,
        CHANNEL,
    )

    usrp.set_rx_freq(

        uhd.types.TuneRequest(
            FREQUENCY_HZ
        ),

        CHANNEL,
    )

    time.sleep(
        SETTLING_TIME_S
    )

    actual_frequency_hz = float(
        usrp.get_rx_freq(
            CHANNEL
        )
    )

    actual_gain_db = float(
        usrp.get_rx_gain(
            CHANNEL
        )
    )

    if (
        abs(
            actual_gain_db
            - GAIN_DB
        )
        > GAIN_TOLERANCE_DB
    ):

        raise RuntimeError(
            f"Gain demandé="
            f"{GAIN_DB:.1f}, "
            f"gain réel="
            f"{actual_gain_db:.1f} dB"
        )

    return (
        actual_frequency_hz,
        actual_gain_db,
    )


# ============================================================
# ACQUISITION
# ============================================================

def acquire_samples(
    usrp: uhd.usrp.MultiUSRP,
) -> np.ndarray:

    number_of_samples = int(
        ACQUISITION_DURATION_S
        * RATE_HZ
    )

    stream_args = (
        uhd.usrp.StreamArgs(
            "fc32",
            "sc16",
        )
    )

    stream_args.channels = [
        CHANNEL
    ]

    streamer = (
        usrp.get_rx_stream(
            stream_args
        )
    )

    metadata = (
        uhd.types.RXMetadata()
    )

    samples = np.zeros(
        number_of_samples,
        dtype=np.complex64,
    )

    buffer = np.zeros(
        (
            1,
            streamer.get_max_num_samps(),
        ),
        dtype=np.complex64,
    )

    command = (
        uhd.types.StreamCMD(
            uhd.types.StreamMode.num_done
        )
    )

    command.num_samps = (
        number_of_samples
    )

    command.stream_now = True

    streamer.issue_stream_cmd(
        command
    )

    total_received = 0

    while (
        total_received
        < number_of_samples
    ):

        received = streamer.recv(
            buffer,
            metadata,
            timeout=3.0,
        )

        if (
            metadata.error_code
            !=
            uhd.types.RXMetadataErrorCode.none
        ):

            raise RuntimeError(
                "Erreur UHD RX : "
                f"{metadata.strerror()}"
            )

        if received <= 0:

            raise RuntimeError(
                "Aucun échantillon reçu"
            )

        count = min(

            received,

            number_of_samples
            - total_received,
        )

        samples[
            total_received:
            total_received + count
        ] = buffer[
            0,
            :count
        ]

        total_received += count

    discard_samples = int(
        DISCARD_DURATION_S
        * RATE_HZ
    )

    samples = samples[
        discard_samples:
    ]

    if samples.size == 0:

        raise RuntimeError(
            "Aucun échantillon utile"
        )

    return samples


# ============================================================
# CALCUL MAX dBFS
# ============================================================

def calculate_max_dbfs(
    samples: np.ndarray,
    lowpass_sos: np.ndarray,
) -> tuple[float, float]:

    """
    Même traitement que le spectre :

        samples
            ↓
        LPF 4 MHz
            ↓
        suppression des bords du filtre
            ↓
        envelope = abs(samples)
            ↓
        max_amplitude
            ↓
        max_dbfs = 20 log10(max_amplitude)

    Retourne :
        max_dbfs
        max_amplitude
    """

    samples = np.asarray(
        samples,
        dtype=np.complex64,
    ).ravel()

    if samples.size == 0:

        raise RuntimeError(
            "Acquisition vide"
        )


    # --------------------------------------------------------
    # Filtre passe-bas
    # --------------------------------------------------------

    filtered_samples = (
        sosfiltfilt(
            lowpass_sos,
            samples,
        )
    )


    # --------------------------------------------------------
    # Suppression des bords du filtre
    # --------------------------------------------------------

    edge_samples = int(
        FILTER_EDGE_DURATION_S
        * RATE_HZ
    )

    if (
        edge_samples > 0
        and
        filtered_samples.size
        > 2 * edge_samples
    ):

        filtered_samples = (
            filtered_samples[
                edge_samples:
                -edge_samples
            ]
        )


    if filtered_samples.size == 0:

        raise RuntimeError(
            "Aucun échantillon "
            "après suppression "
            "des bords du filtre"
        )


    # --------------------------------------------------------
    # Enveloppe
    # --------------------------------------------------------

    envelope = np.abs(
        filtered_samples
    )


    # --------------------------------------------------------
    # MAX
    # --------------------------------------------------------

    max_amplitude = float(
        np.max(
            envelope
        )
    )


    # --------------------------------------------------------
    # Amplitude -> dBFS
    # --------------------------------------------------------

    max_dbfs = float(

        20.0
        * np.log10(
            max(
                max_amplitude,
                EPSILON,
            )
        )
    )

    return (
        max_dbfs,
        max_amplitude,
    )


# ============================================================
# MESURE DE PLUSIEURS ACQUISITIONS
# ============================================================

def measure_max_dbfs(
    usrp: uhd.usrp.MultiUSRP,
    lowpass_sos: np.ndarray,
):

    max_dbfs_values = []

    max_amplitude_values = []

    for _ in range(
        REPEATS_PER_POWER
    ):

        samples = acquire_samples(
            usrp
        )

        (
            max_dbfs,
            max_amplitude,
        ) = calculate_max_dbfs(

            samples,

            lowpass_sos,
        )

        max_dbfs_values.append(
            max_dbfs
        )

        max_amplitude_values.append(
            max_amplitude
        )

    max_dbfs_values = np.asarray(
        max_dbfs_values,
        dtype=float,
    )

    max_amplitude_values = np.asarray(
        max_amplitude_values,
        dtype=float,
    )

    # Même logique que ton spectre :
    # médiane des MAX des différentes acquisitions

    median_max_dbfs = float(
        np.median(
            max_dbfs_values
        )
    )

    std_max_dbfs = float(
        np.std(
            max_dbfs_values
        )
    )

    median_max_amplitude = float(
        np.median(
            max_amplitude_values
        )
    )

    return (
        median_max_dbfs,
        std_max_dbfs,
        median_max_amplitude,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "VALIDATION CALIBRATION B200 "
        "- MAX TEMPOREL"
    )

    print(
        "=" * 80
    )

    print(
        f"Fréquence : "
        f"{FREQUENCY_HZ/1e6:.3f} MHz"
    )

    print(
        f"Gain RX   : "
        f"{GAIN_DB:.1f} dB"
    )

    print(
        f"Rate      : "
        f"{RATE_HZ/1e6:.3f} Msps"
    )

    print(
        f"LPF       : "
        f"{LOWPASS_CUTOFF_HZ/1e6:.2f} MHz"
    )

    print(
        f"N5183A    : "
        f"{GENERATOR_MIN_DBM:.1f} -> "
        f"{GENERATOR_MAX_DBM:.1f} dBm"
    )

    print(
        f"ATT       : "
        f"{ATTENUATION_DB:.1f} dB"
    )

    print(
        f"RX2       : "
        f"{GENERATOR_MIN_DBM - ATTENUATION_DB:.1f} -> "
        f"{GENERATOR_MAX_DBM - ATTENUATION_DB:.1f} dBm"
    )


    # ========================================================
    # CHARGEMENT CALIBRATION
    # ========================================================

    calibration_table = (
        load_calibration()
    )

    (
        selected_gain,
        calibration_frequencies,
        calibration_references,
    ) = build_gain_calibration(

        calibration_table,

        GAIN_DB,
    )

    reference_dbm = (
        get_reference_dbm(

            calibration_frequencies,

            calibration_references,

            FREQUENCY_HZ,
        )
    )

    print()

    print(
        f"Gain calibration : "
        f"{selected_gain:.1f} dB"
    )

    print(
        f"Référence calib   : "
        f"{reference_dbm:.3f} dBm"
    )


    # ========================================================
    # INITIALISATION USRP
    # ========================================================

    print()

    print(
        "Initialisation USRP..."
    )

    usrp = uhd.usrp.MultiUSRP(
        f"serial={USRP_SERIAL}"
    )


    # ========================================================
    # CREATION FILTRE
    # ========================================================

    lowpass_sos = (
        create_lowpass_filter(
            RATE_HZ
        )
    )


    # ========================================================
    # CONFIGURATION USRP
    # ========================================================

    (
        actual_frequency_hz,
        actual_gain_db,
    ) = configure_usrp(
        usrp
    )


    # ========================================================
    # REFERENCE AVEC FREQUENCE REELLE
    # ========================================================

    reference_dbm = (
        get_reference_dbm(

            calibration_frequencies,

            calibration_references,

            actual_frequency_hz,
        )
    )

    print(
        f"Fréquence réelle : "
        f"{actual_frequency_hz/1e6:.6f} MHz"
    )

    print(
        f"Gain réel        : "
        f"{actual_gain_db:.2f} dB"
    )

    print(
        f"Référence utilisée : "
        f"{reference_dbm:.3f} dBm"
    )


    # ========================================================
    # GENERATEUR
    # ========================================================

    generator = N5183A(
        GENERATOR_IP
    )

    generator.output(
        False
    )

    actual_gen_frequency = (
        generator.set_frequency(
            FREQUENCY_HZ
        )
    )

    print(
        f"Fréquence N5183A : "
        f"{actual_gen_frequency/1e6:.6f} MHz"
    )

    print()

    print(
        f"Vérifie qu'il y a EXACTEMENT "
        f"{ATTENUATION_DB:.0f} dB "
        "entre N5183A et RX2."
    )

    answer = input(
        "Tape exactement OUI "
        "pour lancer : "
    ).strip().upper()

    if answer != "OUI":

        generator.close()

        raise RuntimeError(
            "Test annulé"
        )


    # ========================================================
    # NIVEAUX GENERATEUR
    # ========================================================

    generator_levels_dbm = np.arange(

        GENERATOR_MIN_DBM,

        GENERATOR_MAX_DBM
        + GENERATOR_STEP_DB / 2.0,

        GENERATOR_STEP_DB,

        dtype=float,
    )


    # ========================================================
    # RESULTATS
    # ========================================================

    injected_rx2_values = []

    measured_max_dbfs_values = []

    calibrated_dbm_values = []

    error_values = []


    # ========================================================
    # MESURES
    # ========================================================

    try:

        for (
            index,
            requested_gen_dbm,
        ) in enumerate(

            generator_levels_dbm,

            start=1,
        ):

            requested_gen_dbm = float(
                requested_gen_dbm
            )


            # ------------------------------------------------
            # Coupe RF avant changement puissance
            # ------------------------------------------------

            generator.output(
                False
            )


            # ------------------------------------------------
            # Configure puissance
            # ------------------------------------------------

            actual_gen_dbm = (
                generator.set_power(
                    requested_gen_dbm
                )
            )


            # ------------------------------------------------
            # Puissance réelle attendue RX2
            # ------------------------------------------------

            actual_rx2_dbm = (
                actual_gen_dbm
                - ATTENUATION_DB
            )


            # ------------------------------------------------
            # Active RF
            # ------------------------------------------------

            generator.output(
                True
            )

            time.sleep(
                SETTLING_TIME_S
            )


            # ------------------------------------------------
            # MESURE MAX
            # ------------------------------------------------

            (
                measured_max_dbfs,
                measured_std_db,
                max_amplitude,
            ) = measure_max_dbfs(

                usrp,

                lowpass_sos,
            )


            # ------------------------------------------------
            # CONVERSION AVEC REFERENCE ACTUELLE
            # ------------------------------------------------

            calibrated_dbm = (
                measured_max_dbfs
                + reference_dbm
            )


            # ------------------------------------------------
            # ERREUR
            # ------------------------------------------------

            error_db = (
                calibrated_dbm
                - actual_rx2_dbm
            )


            # ------------------------------------------------
            # STOCKAGE
            # ------------------------------------------------

            injected_rx2_values.append(
                actual_rx2_dbm
            )

            measured_max_dbfs_values.append(
                measured_max_dbfs
            )

            calibrated_dbm_values.append(
                calibrated_dbm
            )

            error_values.append(
                error_db
            )


            # ------------------------------------------------
            # AFFICHAGE
            # ------------------------------------------------

            print(
                f"[{index:2d}/"
                f"{len(generator_levels_dbm)}] "

                f"Pgen="
                f"{actual_gen_dbm:6.1f} dBm | "

                f"RX2="
                f"{actual_rx2_dbm:7.1f} dBm | "

                f"MAX amp="
                f"{max_amplitude:.6f} | "

                f"MAX="
                f"{measured_max_dbfs:7.2f} dBFS | "

                f"Calibré="
                f"{calibrated_dbm:7.2f} dBm | "

                f"Erreur="
                f"{error_db:+6.2f} dB | "

                f"σ="
                f"{measured_std_db:.3f} dB"
            )


    finally:

        generator.close()


    # ========================================================
    # CONVERSION NUMPY
    # ========================================================

    injected_rx2_values = np.asarray(
        injected_rx2_values,
        dtype=float,
    )

    measured_max_dbfs_values = np.asarray(
        measured_max_dbfs_values,
        dtype=float,
    )

    calibrated_dbm_values = np.asarray(
        calibrated_dbm_values,
        dtype=float,
    )

    error_values = np.asarray(
        error_values,
        dtype=float,
    )


    # ========================================================
    # RESUME
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "RÉSUMÉ MAX TEMPOREL"
    )

    print(
        "=" * 80
    )

    print(
        f"Erreur moyenne         : "
        f"{np.mean(error_values):+.3f} dB"
    )

    print(
        f"Erreur médiane         : "
        f"{np.median(error_values):+.3f} dB"
    )

    print(
        f"Erreur absolue moyenne : "
        f"{np.mean(np.abs(error_values)):.3f} dB"
    )

    print(
        f"Erreur absolue max     : "
        f"{np.max(np.abs(error_values)):.3f} dB"
    )


    # ========================================================
    # GRAPHE 1
    # RX2 INJECTE VS MAX CALIBRE
    # ========================================================

    plt.figure(
        figsize=(11, 8)
    )

    plt.plot(

        injected_rx2_values,

        calibrated_dbm_values,

        marker="o",

        linewidth=1.7,

        label=(
            "MAX temporel "
            "après calibration"
        ),
    )


    minimum = float(
        min(
            injected_rx2_values.min(),
            calibrated_dbm_values.min(),
        )
    )

    maximum = float(
        max(
            injected_rx2_values.max(),
            calibrated_dbm_values.max(),
        )
    )


    plt.plot(

        [minimum, maximum],

        [minimum, maximum],

        linestyle="--",

        linewidth=2,

        label=(
            "Idéal : "
            "calculé = injecté"
        ),
    )


    plt.title(

        "dBm injecté vs MAX temporel calibré\n"

        f"{actual_frequency_hz/1e6:.1f} MHz - "

        f"Gain {actual_gain_db:.0f} dB"
    )


    plt.xlabel(
        "Puissance injectée "
        "à RX2 (dBm)"
    )


    plt.ylabel(
        "MAX temporel "
        "après calibration (dBm)"
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.legend()

    plt.tight_layout()

    plt.show()


    # ========================================================
    # GRAPHE 2
    # ERREUR
    # ========================================================

    plt.figure(
        figsize=(11, 6)
    )


    plt.plot(

        injected_rx2_values,

        error_values,

        marker="o",

        linewidth=1.7,
    )


    plt.axhline(

        0.0,

        linestyle="--",

        linewidth=2,

        label=(
            "Erreur idéale = 0 dB"
        ),
    )


    plt.axhline(

        +1.0,

        linestyle=":",

        linewidth=1.3,

        label="±1 dB",
    )


    plt.axhline(

        -1.0,

        linestyle=":",

        linewidth=1.3,
    )


    plt.title(

        "Erreur calibration MAX temporel\n"

        f"{actual_frequency_hz/1e6:.1f} MHz - "

        f"Gain {actual_gain_db:.0f} dB"
    )


    plt.xlabel(
        "Puissance injectée "
        "à RX2 (dBm)"
    )


    plt.ylabel(
        "Erreur = "
        "MAX calibré - injecté (dB)"
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.legend()

    plt.tight_layout()

    plt.show()


    # ========================================================
    # GRAPHE 3
    # MAX dBFS VS PUISSANCE RX2
    # ========================================================

    plt.figure(
        figsize=(11, 7)
    )


    plt.plot(

        injected_rx2_values,

        measured_max_dbfs_values,

        marker="o",

        linewidth=1.7,
    )


    plt.title(

        "MAX temporel dBFS en fonction "
        "de la puissance injectée\n"

        f"{actual_frequency_hz/1e6:.1f} MHz - "

        f"Gain {actual_gain_db:.0f} dB"
    )


    plt.xlabel(
        "Puissance injectée "
        "à RX2 (dBm)"
    )


    plt.ylabel(
        "MAX temporel (dBFS)"
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.tight_layout()

    plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()