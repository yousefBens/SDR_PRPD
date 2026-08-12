#!/usr/bin/env python3
from __future__ import annotations

import math
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvisa
import uhd


# ============================================================
# CONFIGURATION
# ============================================================

USRP_SERIAL = "306BD15"
CHANNEL = 0
ANTENNA = "RX2"

# Fréquence et gain à tester
FREQUENCY_HZ = 1.0e9
GAIN_DB = 60.0

# Même sample rate que la calibration
RATE_HZ = 12e6

# Acquisition
DISCARD_DURATION_S = 0.002
USEFUL_DURATION_S = 0.020
ACQUISITION_DURATION_S = DISCARD_DURATION_S + USEFUL_DURATION_S
SETTLING_TIME_S = 0.15
REPEATS_PER_POWER = 5

# Générateur
GENERATOR_IP = "192.168.1.100"
GENERATOR_MIN_DBM = -20.0
GENERATOR_MAX_DBM = +20.0
GENERATOR_STEP_DB = 1.0

# Atténuation physique
ATTENUATION_DB = 60.0

# Limite sûre côté RX2
RX2_SAFE_MAX_DBM = -20.0

# Calibration
CAL_FILE = Path(
    "/home/yousef/Documents/testing_scripts/GEVernova/"
    "Fi_Wo_Ca_Dy/Calib/Cal_Files/"
    "rx2_power_calibration_final.pickle"
)

GAIN_TOLERANCE_DB = 0.5
FREQUENCY_TOLERANCE_HZ = 1e3
EPSILON = 1e-20


# ============================================================
# N5183A
# ============================================================

class N5183A:
    def __init__(self, ip: str):
        self.rm = pyvisa.ResourceManager("@py")

        resources = [
            f"TCPIP0::{ip}::inst0::INSTR",
            f"TCPIP0::{ip}::5025::SOCKET",
        ]

        self.dev = None
        last_error = None

        for resource in resources:
            print(f"[N5183A] Essai : {resource}")
            dev = None

            try:
                dev = self.rm.open_resource(
                    resource,
                    open_timeout=5000,
                )
                dev.timeout = 10000
                dev.write_termination = "\n"
                dev.read_termination = "\n"

                identity = dev.query("*IDN?").strip()
                print(f"[N5183A] {identity}")

                if "N5183A" not in identity.upper():
                    raise RuntimeError(
                        f"Instrument inattendu : {identity}"
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
                f"Impossible de connecter le N5183A : {last_error}"
            )

        print(f"[N5183A] Ressource : {self.resource}")

        self.dev.write(":OUTP OFF")
        self.dev.write(":FREQ:MODE FIX")
        self.dev.write(":POW:MODE FIX")
        self._check_errors("initialisation")

    def output(self, enabled: bool):
        self.dev.write(":OUTP ON" if enabled else ":OUTP OFF")
        state = int(float(self.dev.query(":OUTP?")))

        if state != int(enabled):
            raise RuntimeError("Échec changement état RF")

        self._check_errors("RF output")

    def set_frequency(self, frequency_hz: float) -> float:
        frequency_hz = float(frequency_hz)

        self.dev.write(
            f":FREQ {frequency_hz:.3f} HZ"
        )

        actual = float(
            self.dev.query(":FREQ?")
        )

        self._check_errors("fréquence")
        return actual

    def set_power(self, power_dbm: float) -> float:
        power_dbm = float(power_dbm)

        if not GENERATOR_MIN_DBM <= power_dbm <= GENERATOR_MAX_DBM:
            raise RuntimeError(
                f"Puissance N5183A interdite : {power_dbm:.2f} dBm"
            )

        expected_rx2_dbm = (
            power_dbm - ATTENUATION_DB
        )

        if expected_rx2_dbm > RX2_SAFE_MAX_DBM:
            raise RuntimeError(
                "ARRÊT SÉCURITÉ : "
                f"RX2 recevrait {expected_rx2_dbm:.2f} dBm"
            )

        self.dev.write(
            f":POW {power_dbm:.3f} DBM"
        )

        actual = float(
            self.dev.query(":POW?")
        )

        self._check_errors("puissance")

        if abs(actual - power_dbm) > 0.10:
            self.output(False)
            raise RuntimeError(
                f"Pgen demandée={power_dbm:.3f} dBm, "
                f"réelle={actual:.3f} dBm"
            )

        return actual

    def _check_errors(self, operation: str):
        errors = []

        while True:
            response = self.dev.query(
                "SYST:ERR?"
            ).strip()

            if response.startswith("+0") or response.startswith("0"):
                break

            errors.append(response)

        if errors:
            raise RuntimeError(
                f"Erreur SCPI ({operation}) : "
                + " | ".join(errors)
            )

    def close(self):
        try:
            if self.dev is not None:
                try:
                    self.dev.write(":OUTP OFF")
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
# CALIBRATION
# ============================================================

def load_calibration() -> dict:
    if not CAL_FILE.exists():
        raise FileNotFoundError(
            f"Calibration introuvable : {CAL_FILE}"
        )

    with CAL_FILE.open("rb") as f:
        calibration = pickle.load(f)

    if CHANNEL not in calibration:
        raise ValueError(
            f"Canal {CHANNEL} absent de la calibration"
        )

    if ANTENNA not in calibration[CHANNEL]:
        raise ValueError(
            f"Antenne {ANTENNA} absente de la calibration"
        )

    raw_table = calibration[
        CHANNEL
    ][
        ANTENNA
    ]

    table = {}

    for freq_hz, gain_table in raw_table.items():
        table[float(freq_hz)] = {
            float(gain_db): float(reference_dbm)
            for gain_db, reference_dbm
            in gain_table.items()
        }

    return table


def select_gain(
    calibration_table: dict,
    requested_gain_db: float,
) -> float:
    gains = sorted(
        {
            float(gain_db)
            for gain_table in calibration_table.values()
            for gain_db in gain_table.keys()
        }
    )

    if not gains:
        raise ValueError(
            "Aucun gain dans la calibration"
        )

    gains = np.asarray(
        gains,
        dtype=float,
    )

    selected_gain = float(
        gains[
            np.argmin(
                np.abs(
                    gains - requested_gain_db
                )
            )
        ]
    )

    if abs(selected_gain - requested_gain_db) > GAIN_TOLERANCE_DB:
        raise ValueError(
            f"Gain demandé={requested_gain_db:.1f} dB, "
            f"gain calibré le plus proche={selected_gain:.1f} dB"
        )

    return selected_gain


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
        gain_table = calibration_table[
            frequency_hz
        ]

        if selected_gain not in gain_table:
            continue

        frequencies.append(
            float(frequency_hz)
        )

        references.append(
            float(
                gain_table[selected_gain]
            )
        )

    if not frequencies:
        raise RuntimeError(
            f"Aucune calibration pour gain {selected_gain:.1f} dB"
        )

    return (
        selected_gain,
        np.asarray(frequencies, dtype=float),
        np.asarray(references, dtype=float),
    )


def get_reference_dbm(
    frequencies_hz: np.ndarray,
    references_dbm: np.ndarray,
    frequency_hz: float,
) -> float:
    frequency_hz = float(frequency_hz)

    fmin = float(
        frequencies_hz.min()
    )

    fmax = float(
        frequencies_hz.max()
    )

    if frequency_hz < fmin:
        if fmin - frequency_hz <= FREQUENCY_TOLERANCE_HZ:
            frequency_hz = fmin
        else:
            raise ValueError(
                f"{frequency_hz/1e6:.3f} MHz sous la plage calibrée"
            )

    if frequency_hz > fmax:
        if frequency_hz - fmax <= FREQUENCY_TOLERANCE_HZ:
            frequency_hz = fmax
        else:
            raise ValueError(
                f"{frequency_hz/1e6:.3f} MHz au-dessus de la plage calibrée"
            )

    return float(
        np.interp(
            frequency_hz,
            frequencies_hz,
            references_dbm,
        )
    )


# ============================================================
# USRP
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
        usrp.get_rx_freq(CHANNEL)
    )

    actual_gain_db = float(
        usrp.get_rx_gain(CHANNEL)
    )

    if abs(actual_gain_db - GAIN_DB) > GAIN_TOLERANCE_DB:
        raise RuntimeError(
            f"Gain demandé={GAIN_DB:.1f}, "
            f"gain réel={actual_gain_db:.1f} dB"
        )

    return (
        actual_frequency_hz,
        actual_gain_db,
    )


def acquire_samples(
    usrp: uhd.usrp.MultiUSRP,
) -> np.ndarray:
    number_of_samples = int(
        ACQUISITION_DURATION_S
        * RATE_HZ
    )

    stream_args = uhd.usrp.StreamArgs(
        "fc32",
        "sc16",
    )
    stream_args.channels = [CHANNEL]

    streamer = usrp.get_rx_stream(
        stream_args
    )

    metadata = uhd.types.RXMetadata()

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

    command = uhd.types.StreamCMD(
        uhd.types.StreamMode.num_done
    )

    command.num_samps = number_of_samples
    command.stream_now = True

    streamer.issue_stream_cmd(
        command
    )

    total_received = 0

    while total_received < number_of_samples:
        received = streamer.recv(
            buffer,
            metadata,
            timeout=3.0,
        )

        if (
            metadata.error_code
            != uhd.types.RXMetadataErrorCode.none
        ):
            raise RuntimeError(
                f"Erreur UHD RX : {metadata.strerror()}"
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

    return samples[
        discard_samples:
    ]


# ============================================================
# MESURE dBFS
# ============================================================

def power_dbfs(
    samples: np.ndarray,
) -> float:
    """
    Puissance moyenne numérique complexe :

        dBFS = 10*log10(mean(|x|²))
    """

    samples = np.asarray(
        samples,
        dtype=np.complex64,
    ).ravel()

    if samples.size == 0:
        raise RuntimeError(
            "Acquisition vide"
        )

    power = float(
        np.mean(
            np.abs(samples) ** 2
        )
    )

    return float(
        10.0
        * math.log10(
            max(
                power,
                EPSILON,
            )
        )
    )


def measure_dbfs(
    usrp: uhd.usrp.MultiUSRP,
):
    values = []

    for _ in range(
        REPEATS_PER_POWER
    ):
        values.append(
            power_dbfs(
                acquire_samples(
                    usrp
                )
            )
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(
            np.median(values)
        ),
        float(
            np.std(values)
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("VALIDATION CALIBRATION B200 - UNE FREQUENCE")
    print("=" * 80)

    print(
        f"Fréquence : {FREQUENCY_HZ/1e6:.3f} MHz"
    )
    print(
        f"Gain RX   : {GAIN_DB:.1f} dB"
    )
    print(
        f"N5183A    : {GENERATOR_MIN_DBM:.1f} -> "
        f"{GENERATOR_MAX_DBM:.1f} dBm"
    )
    print(
        f"ATT       : {ATTENUATION_DB:.1f} dB"
    )
    print(
        f"RX2       : "
        f"{GENERATOR_MIN_DBM - ATTENUATION_DB:.1f} -> "
        f"{GENERATOR_MAX_DBM - ATTENUATION_DB:.1f} dBm"
    )

    # Calibration
    calibration_table = load_calibration()

    (
        selected_gain,
        calibration_frequencies,
        calibration_references,
    ) = build_gain_calibration(
        calibration_table,
        GAIN_DB,
    )

    reference_dbm = get_reference_dbm(
        calibration_frequencies,
        calibration_references,
        FREQUENCY_HZ,
    )

    print()
    print(
        f"Gain calibration : {selected_gain:.1f} dB"
    )
    print(
        f"Référence calib   : {reference_dbm:.3f} dBm"
    )

    # USRP
    print()
    print("Initialisation USRP...")

    usrp = uhd.usrp.MultiUSRP(
        f"serial={USRP_SERIAL}"
    )

    (
        actual_frequency_hz,
        actual_gain_db,
    ) = configure_usrp(
        usrp
    )

    # Référence avec fréquence réelle UHD
    reference_dbm = get_reference_dbm(
        calibration_frequencies,
        calibration_references,
        actual_frequency_hz,
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

    # Générateur
    generator = N5183A(
        GENERATOR_IP
    )

    generator.output(False)

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
        f"{ATTENUATION_DB:.0f} dB entre N5183A et RX2."
    )

    answer = input(
        "Tape exactement OUI pour lancer : "
    ).strip().upper()

    if answer != "OUI":
        generator.close()
        raise RuntimeError(
            "Test annulé"
        )

    generator_levels_dbm = np.arange(
        GENERATOR_MIN_DBM,
        GENERATOR_MAX_DBM
        + GENERATOR_STEP_DB / 2.0,
        GENERATOR_STEP_DB,
        dtype=float,
    )

    injected_rx2_values = []
    calibrated_dbm_values = []
    error_values = []

    try:
        for index, requested_gen_dbm in enumerate(
            generator_levels_dbm,
            start=1,
        ):
            requested_gen_dbm = float(
                requested_gen_dbm
            )

            generator.output(False)

            actual_gen_dbm = (
                generator.set_power(
                    requested_gen_dbm
                )
            )

            actual_rx2_dbm = (
                actual_gen_dbm
                - ATTENUATION_DB
            )

            generator.output(True)

            time.sleep(
                SETTLING_TIME_S
            )

            (
                measured_dbfs,
                measured_std_db,
            ) = measure_dbfs(
                usrp
            )

            calibrated_dbm = (
                measured_dbfs
                + reference_dbm
            )

            error_db = (
                calibrated_dbm
                - actual_rx2_dbm
            )

            injected_rx2_values.append(
                actual_rx2_dbm
            )

            calibrated_dbm_values.append(
                calibrated_dbm
            )

            error_values.append(
                error_db
            )

            print(
                f"[{index:2d}/{len(generator_levels_dbm)}] "
                f"Pgen={actual_gen_dbm:6.1f} dBm | "
                f"RX2={actual_rx2_dbm:7.1f} dBm | "
                f"Mesure={measured_dbfs:7.2f} dBFS | "
                f"Calibré={calibrated_dbm:7.2f} dBm | "
                f"Erreur={error_db:+6.2f} dB | "
                f"σ={measured_std_db:.3f} dB"
            )

    finally:
        generator.close()

    injected_rx2_values = np.asarray(
        injected_rx2_values,
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

    # Résumé
    print()
    print("=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)

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
    # GRAPHE 1 : injecté vs calibré
    # ========================================================

    plt.figure(
        figsize=(11, 8)
    )

    plt.plot(
        injected_rx2_values,
        calibrated_dbm_values,
        marker="o",
        linewidth=1.7,
        label="dBm calculé après calibration",
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
        label="Idéal : calculé = injecté",
    )

    plt.title(
        "dBm injecté vs dBm calibré\n"
        f"{actual_frequency_hz/1e6:.1f} MHz - "
        f"Gain {actual_gain_db:.0f} dB"
    )

    plt.xlabel(
        "Puissance injectée à RX2 (dBm)"
    )

    plt.ylabel(
        "Puissance calculée après calibration (dBm)"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    # ========================================================
    # GRAPHE 2 : erreur
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
        label="Erreur idéale = 0 dB",
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
        "Erreur de calibration\n"
        f"{actual_frequency_hz/1e6:.1f} MHz - "
        f"Gain {actual_gain_db:.0f} dB"
    )

    plt.xlabel(
        "Puissance injectée à RX2 (dBm)"
    )

    plt.ylabel(
        "Erreur = calibré - injecté (dB)"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()