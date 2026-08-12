#!/usr/bin/env python3
#
# Copyright 2020 Ettus Research, a National Instruments Brand
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Calibration de puissance UHD personnalisée.

Ajouts par rapport à uhd_power_cal.py :
- --gain-min / --gain-max
- --min-input-power
- --target-dbfs
- --dbfs-lower-limit / --dbfs-upper-limit
- création automatique du dossier du fichier --store

Ces options permettent de calibrer séparément plusieurs groupes de gains
avec différentes atténuations et une cible dBFS adaptée.
"""

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import uhd
import uhd.usrp.cal.usrp_calibrator as usrp_calibrator_module


def parse_args():
    """Analyser les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description=(
            "Run power level calibration for supported USRPs "
            "with gain and dBFS limits."
        )
    )

    parser.add_argument(
        "--args",
        default="",
        help="USRP Device Args",
    )

    parser.add_argument(
        "-d",
        "--dir",
        default="tx",
        choices=["tx", "rx"],
        help="Calibration direction: rx or tx.",
    )

    parser.add_argument(
        "--start",
        type=float,
        help="Start frequency in Hz.",
    )

    parser.add_argument(
        "--stop",
        type=float,
        help="Stop frequency in Hz.",
    )

    parser.add_argument(
        "--step",
        type=float,
        help="Frequency step in Hz.",
    )

    parser.add_argument(
        "--gain-step",
        type=float,
        default=1.0,
        help="Gain step in dB.",
    )

    parser.add_argument(
        "--gain-min",
        type=float,
        default=None,
        help="Minimum requested gain to calibrate in dB.",
    )

    parser.add_argument(
        "--gain-max",
        type=float,
        default=None,
        help="Maximum requested gain to calibrate in dB.",
    )

    parser.add_argument(
        "--min-input-power",
        type=float,
        default=None,
        help=(
            "Initial power requested at the DUT input, in dBm. "
            "Overrides the calibrator default min_detectable_signal."
        ),
    )

    parser.add_argument(
        "--target-dbfs",
        type=float,
        default=None,
        help=(
            "Desired digital power during RX calibration in dBFS. "
            "UHD default is -6 dBFS."
        ),
    )

    parser.add_argument(
        "--dbfs-lower-limit",
        type=float,
        default=None,
        help=(
            "Lowest accepted received level in dBFS. "
            "UHD default is -20 dBFS."
        ),
    )

    parser.add_argument(
        "--dbfs-upper-limit",
        type=float,
        default=None,
        help=(
            "Highest accepted received level in dBFS. "
            "UHD default is -3 dBFS."
        ),
    )

    parser.add_argument(
        "--lo-offset",
        type=float,
        help="LO offset in Hz.",
    )

    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0 / math.sqrt(2.0),
        help="TX tone amplitude. Default: 1/sqrt(2).",
    )

    parser.add_argument(
        "--attenuation",
        type=float,
        default=0.0,
        help=(
            "Positive attenuation in dB between the measurement "
            "device and the DUT."
        ),
    )

    parser.add_argument(
        "--tone-freq",
        type=float,
        default=1e6,
        help="TX tone offset frequency in Hz.",
    )

    parser.add_argument(
        "--antenna",
        default="*",
        help="Antenna port, or '*' for all valid ports.",
    )

    parser.add_argument(
        "--channels",
        default="*",
        help="Channel list such as 0 or 0,1, or '*'.",
    )

    parser.add_argument(
        "--meas-dev",
        default="manual",
        help="Measurement device type.",
    )

    parser.add_argument(
        "-o",
        "--meas-option",
        default=[],
        action="append",
        help="Option passed to the measurement device.",
    )

    parser.add_argument(
        "--switch",
        default="manual",
        help="RF switch type.",
    )

    parser.add_argument(
        "--switch-option",
        default=[],
        action="append",
        help="Option passed to the RF switch.",
    )

    parser.add_argument(
        "-r",
        "--rate",
        type=float,
        help="Sampling rate in samples per second.",
    )

    parser.add_argument(
        "--store",
        metavar="filename.pickle",
        help="Store intermediate calibration results in a pickle file.",
    )

    parser.add_argument(
        "--load",
        metavar="filename.pickle",
        help="Load intermediate calibration data instead of measuring.",
    )

    return parser.parse_args()


def sanitize_args(usrp, args, default_rate):
    """Valider canaux, antennes et fréquence d'échantillonnage."""
    if usrp.get_num_mboards() != 1:
        raise RuntimeError(
            "Power calibration tools support one motherboard only."
        )

    get_num_channels = getattr(usrp, f"get_{args.dir}_num_channels")
    available_channels = get_num_channels()

    if args.channels == "*":
        channels = list(range(available_channels))
    else:
        try:
            channels = [int(value) for value in args.channels.split(",")]
        except ValueError as error:
            raise ValueError(
                f"Invalid channel list: {args.channels}"
            ) from error

        for channel in channels:
            if channel not in range(available_channels):
                raise ValueError(
                    f"Invalid channel {channel}; expected 0 to "
                    f"{available_channels - 1}."
                )

    print(
        "=== Calibrating for channels:",
        ", ".join(str(channel) for channel in channels),
    )

    get_antennas = getattr(usrp, f"get_{args.dir}_antennas")
    available_antennas = get_antennas()

    if args.antenna == "*":
        invalid_antennas = (
            "CAL",
            "LOCAL",
            "CAL_LOOPBACK",
            "TERMINATION",
        )
        antennas = [
            antenna
            for antenna in available_antennas
            if antenna not in invalid_antennas
        ]
    else:
        antennas = args.antenna.split(",")
        for antenna in antennas:
            if antenna not in available_antennas:
                raise ValueError(
                    f"Invalid antenna {antenna}; "
                    f"available antennas: {available_antennas}"
                )

    print(
        "=== Calibrating for antennas:",
        ", ".join(antennas),
    )

    rate = args.rate or default_rate
    set_rate = getattr(usrp, f"set_{args.dir}_rate")
    get_rate = getattr(usrp, f"get_{args.dir}_rate")

    set_rate(rate)
    actual_rate = get_rate()

    print(
        f"=== Requested sampling rate: {rate / 1e6:.3f} Msps, "
        f"actual rate: {actual_rate / 1e6:.3f} Msps"
    )

    if args.dir == "tx" and abs(args.tone_freq) > actual_rate:
        raise ValueError(
            "TX tone frequency offset is greater than the sampling rate."
        )

    return channels, antennas, actual_rate


def init_results(pickle_file):
    """Charger les résultats pickle ou retourner un dictionnaire vide."""
    if pickle_file is None:
        return {}

    with open(Path(pickle_file).expanduser(), "rb") as results_file:
        return pickle.load(results_file)


def configure_dbfs_limits(args):
    """Modifier les constantes utilisées par run_rx_cal()."""
    if args.target_dbfs is not None:
        usrp_calibrator_module.PWR_EST_IDEAL_LEVEL = float(
            args.target_dbfs
        )

    if args.dbfs_lower_limit is not None:
        usrp_calibrator_module.PWR_EST_LLIM = float(
            args.dbfs_lower_limit
        )

    if args.dbfs_upper_limit is not None:
        usrp_calibrator_module.PWR_EST_ULIM = float(
            args.dbfs_upper_limit
        )

    target = float(usrp_calibrator_module.PWR_EST_IDEAL_LEVEL)
    lower = float(usrp_calibrator_module.PWR_EST_LLIM)
    upper = float(usrp_calibrator_module.PWR_EST_ULIM)

    if lower > upper:
        raise ValueError(
            "--dbfs-lower-limit cannot be greater than "
            "--dbfs-upper-limit."
        )

    if not lower <= target <= upper:
        raise ValueError(
            "--target-dbfs must be inside the accepted dBFS interval."
        )

    print(f"=== dBFS calibration target: {target:.2f} dBFS")
    print(
        f"=== Accepted dBFS interval: "
        f"{lower:.2f} to {upper:.2f} dBFS"
    )


def filter_calibration_gains(
    usrp_cal,
    direction,
    gain_min,
    gain_max,
):
    """Filtrer la liste interne des gains du calibrateur UHD."""
    if not hasattr(usrp_cal, "_gains"):
        raise RuntimeError(
            "This UHD calibrator does not expose the _gains attribute."
        )

    all_gains = [float(gain) for gain in usrp_cal._gains]

    selected_gains = [
        gain
        for gain in all_gains
        if (gain_min is None or gain >= gain_min)
        and (gain_max is None or gain <= gain_max)
    ]

    if not selected_gains:
        raise ValueError(
            "No gain remains after filtering. "
            f"Available gains: {all_gains}; "
            f"requested interval: [{gain_min}, {gain_max}]"
        )

    selected_gains = sorted(
        selected_gains,
        reverse=(direction == "rx"),
    )

    usrp_cal._gains = selected_gains

    print(
        "=== Available calibration gains:",
        ", ".join(f"{gain:.1f}" for gain in all_gains),
    )
    print(
        "=== Selected calibration gains:",
        ", ".join(f"{gain:.1f}" for gain in selected_gains),
    )


class CalRunner:
    """Exécuter la calibration à une fréquence."""

    def __init__(self, usrp, usrp_cal, meas_dev, args):
        self.usrp = usrp
        self.usrp_cal = usrp_cal
        self.meas_dev = meas_dev
        self.direction = args.dir
        self.tone_offset = args.tone_freq if args.dir == "tx" else 0.0
        self.lo_offset = (
            args.lo_offset
            if args.lo_offset is not None
            else usrp_cal.lo_offset
        )

        if self.lo_offset:
            print(
                f"=== Using USRP LO offset: "
                f"{self.lo_offset / 1e6:.2f} MHz"
            )

    def run(self, channel, frequency):
        """Effectuer tous les traitements pour une fréquence."""
        print(
            f"=== Running calibration at frequency "
            f"{frequency / 1e6:.3f} MHz..."
        )

        tune_request = uhd.types.TuneRequest(
            frequency,
            self.lo_offset,
        )

        set_frequency = getattr(
            self.usrp,
            f"set_{self.direction}_freq",
        )
        get_frequency = getattr(
            self.usrp,
            f"get_{self.direction}_freq",
        )

        set_frequency(tune_request, channel)
        time.sleep(self.usrp_cal.tune_settling_time)

        actual_frequency = get_frequency(channel)

        if abs(actual_frequency - frequency) > 1.0:
            print(
                f"WARNING: Frequency coerced from "
                f"{frequency / 1e6:.3f} MHz to "
                f"{actual_frequency / 1e6:.3f} MHz."
            )

        self.meas_dev.set_frequency(
            actual_frequency + self.tone_offset
        )

        run_calibration = getattr(
            self.usrp_cal,
            f"run_{self.direction}_cal",
        )
        run_calibration(frequency)


def main():
    """Programme principal."""
    args = parse_args()

    if (
        args.gain_min is not None
        and args.gain_max is not None
        and args.gain_min > args.gain_max
    ):
        raise ValueError(
            "--gain-min cannot be greater than --gain-max."
        )

    if args.attenuation < 0:
        raise ValueError("--attenuation must be positive.")

    configure_dbfs_limits(args)

    print("=== Detecting USRP...")
    usrp = uhd.usrp.MultiUSRP(args.args)

    print("=== Measurement direction:", args.dir)
    print("=== Initializing measurement device...")

    meas_dev = uhd.usrp.cal.get_meas_device(
        args.dir,
        args.meas_dev,
        args.meas_option,
    )

    # Exemple :
    # entrée RX2 souhaitée = -70 dBm
    # atténuation = 60 dB
    # sortie générateur commandée = -10 dBm
    meas_dev.power_offset = args.attenuation

    if args.dir == "tx":
        meas_dev.power_offset -= (
            20.0 * math.log10(args.amplitude)
        )

    print(
        f"=== Measurement path attenuation: "
        f"{args.attenuation:.2f} dB"
    )

    print("=== Initializing port connector...")
    switch = uhd.usrp.cal.get_switch(
        args.dir,
        args.switch,
        args.switch_option,
    )

    print("=== Initializing USRP calibration object...")
    usrp_cal = uhd.usrp.cal.get_usrp_calibrator(
        usrp,
        meas_dev,
        args.dir,
        gain_step=args.gain_step,
    )

    if args.min_input_power is not None:
        usrp_cal.min_detectable_signal = float(
            args.min_input_power
        )
        print(
            f"=== Initial RX input power: "
            f"{usrp_cal.min_detectable_signal:.2f} dBm"
        )

    filter_calibration_gains(
        usrp_cal=usrp_cal,
        direction=args.dir,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
    )

    channels, antennas, rate = sanitize_args(
        usrp,
        args,
        usrp_cal.default_rate,
    )

    results = init_results(args.load)

    usrp_cal.init(
        rate=rate,
        tone_freq=args.tone_freq,
        amplitude=args.amplitude,
    )

    print("=== Launching calibration...")
    cal_runner = CalRunner(
        usrp,
        usrp_cal,
        meas_dev,
        args,
    )

    for channel in channels:
        if channel not in results:
            results[channel] = {}

        for antenna in antennas:
            if antenna in results[channel]:
                print(
                    f"=== Using pickled data for channel "
                    f"{channel}, antenna {antenna}."
                )
                continue

            print(
                f"=== Running calibration for channel "
                f"{channel}, antenna {antenna}."
            )

            set_antenna = getattr(
                usrp,
                f"set_{args.dir}_antenna",
            )
            set_antenna(antenna, channel)

            switch.connect(channel, antenna)
            usrp_cal.update_port(channel, antenna)

            frequencies = usrp_cal.init_frequencies(
                args.start,
                args.stop,
                args.step,
            )

            usrp_cal.start()

            try:
                for frequency in frequencies:
                    cal_runner.run(channel, frequency)
            except (
                RuntimeError,
                ValueError,
                KeyboardInterrupt,
            ) as error:
                print(
                    f"ERROR: Stopping calibration due to "
                    f"exception: {error}"
                )
                usrp_cal.stop(store=False)
                return 1

            results[channel][antenna] = usrp_cal.results
            usrp_cal.stop()

    if args.store:
        output_path = Path(args.store).expanduser()
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"=== Storing pickled calibration data to "
            f"{output_path}..."
        )

        with output_path.open("wb") as results_file:
            pickle.dump(results, results_file)

    print("=== Calibration completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as error:
        print("ERROR:", str(error))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nERROR: Calibration interrupted by user.")
        sys.exit(130)
