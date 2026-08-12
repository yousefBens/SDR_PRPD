#!/usr/bin/env python3
# Based on Ettus Research / NI uhd_power_cal.py
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import uhd
import uhd.usrp.cal.usrp_calibrator as cal_module

B200_RX_MAX_GAIN_DB = 76.0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--args", default="")
    parser.add_argument("-d", "--dir", default="rx", choices=["rx", "tx"])
    parser.add_argument("--start", type=float)
    parser.add_argument("--stop", type=float)
    parser.add_argument("--step", type=float)
    parser.add_argument("--gain-step", type=float, default=1.0)

    parser.add_argument(
        "--gains",
        default=None,
        help="Explicit comma-separated RX gains, e.g. 0,20,40,60,76",
    )

    parser.add_argument("--min-input-power", type=float, default=None)
    parser.add_argument("--target-dbfs", type=float, default=None)
    parser.add_argument("--dbfs-lower-limit", type=float, default=None)
    parser.add_argument("--dbfs-upper-limit", type=float, default=None)

    parser.add_argument("--lo-offset", type=float)
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0 / math.sqrt(2.0),
    )
    parser.add_argument("--attenuation", type=float, default=0.0)
    parser.add_argument("--tone-freq", type=float, default=1e6)
    parser.add_argument("--antenna", default="*")
    parser.add_argument("--channels", default="*")
    parser.add_argument("--meas-dev", default="manual")
    parser.add_argument("-o", "--meas-option", default=[], action="append")
    parser.add_argument("--switch", default="manual")
    parser.add_argument("--switch-option", default=[], action="append")
    parser.add_argument("-r", "--rate", type=float)
    parser.add_argument("--store", metavar="filename.pickle")
    parser.add_argument("--load", metavar="filename.pickle")

    return parser.parse_args()


def configure_calibrator_limits(args):
    if args.target_dbfs is not None:
        cal_module.PWR_EST_IDEAL_LEVEL = float(args.target_dbfs)

    if args.dbfs_lower_limit is not None:
        cal_module.PWR_EST_LLIM = float(args.dbfs_lower_limit)

    if args.dbfs_upper_limit is not None:
        cal_module.PWR_EST_ULIM = float(args.dbfs_upper_limit)

    if args.dir == "rx":
        target = float(cal_module.PWR_EST_IDEAL_LEVEL)
        lower = float(cal_module.PWR_EST_LLIM)
        upper = float(cal_module.PWR_EST_ULIM)

        if lower > upper:
            raise ValueError("dBFS lower limit must be <= upper limit")

        if not lower <= target <= upper:
            raise ValueError("target dBFS must be inside accepted dBFS limits")

        print(f"=== RX target level: {target:.2f} dBFS")
        print(
            f"=== RX accepted interval: "
            f"{lower:.2f} -> {upper:.2f} dBFS"
        )


def sanitize_args(usrp, args, default_rate):
    if usrp.get_num_mboards() != 1:
        raise RuntimeError("Power calibration supports one motherboard only")

    available_channels = getattr(
        usrp,
        f"get_{args.dir}_num_channels",
    )()

    if args.channels == "*":
        channels = list(range(available_channels))
    else:
        channels = [int(v) for v in args.channels.split(",")]

    for channel in channels:
        if channel not in range(available_channels):
            raise ValueError(f"Invalid channel {channel}")

    available_antennas = getattr(
        usrp,
        f"get_{args.dir}_antennas",
    )()

    if args.antenna == "*":
        invalid = {"CAL", "LOCAL", "CAL_LOOPBACK", "TERMINATION"}
        antennas = [
            ant
            for ant in available_antennas
            if ant not in invalid
        ]
    else:
        antennas = args.antenna.split(",")

    for antenna in antennas:
        if antenna not in available_antennas:
            raise ValueError(
                f"Invalid antenna {antenna}; available: {available_antennas}"
            )

    rate = args.rate or default_rate

    getattr(
        usrp,
        f"set_{args.dir}_rate",
    )(rate)

    actual_rate = getattr(
        usrp,
        f"get_{args.dir}_rate",
    )()

    print(f"=== Channels: {channels}")
    print(f"=== Antennas: {antennas}")
    print(
        f"=== Requested sample rate: "
        f"{rate/1e6:.3f} Msps, actual: {actual_rate/1e6:.3f} Msps"
    )

    return channels, antennas, actual_rate


def init_results(pickle_file):
    if pickle_file is None:
        return {}

    path = Path(pickle_file).expanduser()

    with path.open("rb") as file:
        return pickle.load(file)


def get_real_rx_gain_max(usrp, channel):
    gain_range = usrp.get_rx_gain_range(channel)

    try:
        value = float(gain_range.stop())
    except TypeError:
        value = float(gain_range.stop)

    return min(value, B200_RX_MAX_GAIN_DB)


def configure_explicit_gains(usrp, usrp_cal, args, channel):
    if args.gains is None:
        return

    if args.dir != "rx":
        raise ValueError("--gains explicit handling is intended for RX")

    if not hasattr(usrp_cal, "_gains"):
        raise RuntimeError("This UHD version does not expose calibrator._gains")

    requested = [float(v) for v in args.gains.split(",")]
    real_max = get_real_rx_gain_max(usrp, channel)

    selected = []

    for gain in requested:
        if gain < 0:
            raise ValueError(f"Invalid gain {gain:.1f} dB")

        if gain > real_max + 1e-6:
            raise ValueError(
                f"Requested gain {gain:.1f} dB exceeds real max "
                f"{real_max:.1f} dB"
            )

        selected.append(gain)

    selected = sorted(set(selected), reverse=True)
    usrp_cal._gains = selected

    print(f"=== Real RX maximum gain: {real_max:.1f} dB")
    print(
        "=== Exact RX calibration gains: "
        + ", ".join(f"{g:.1f}" for g in selected)
    )


class CalRunner:
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

    def run(self, channel, frequency):
        print(
            f"=== Running calibration at "
            f"{frequency/1e6:.3f} MHz..."
        )

        tune_request = uhd.types.TuneRequest(
            frequency,
            self.lo_offset,
        )

        getattr(
            self.usrp,
            f"set_{self.direction}_freq",
        )(
            tune_request,
            channel,
        )

        time.sleep(self.usrp_cal.tune_settling_time)

        actual_frequency = getattr(
            self.usrp,
            f"get_{self.direction}_freq",
        )(channel)

        self.meas_dev.set_frequency(
            actual_frequency + self.tone_offset
        )

        getattr(
            self.usrp_cal,
            f"run_{self.direction}_cal",
        )(frequency)


def main():
    args = parse_args()

    if args.attenuation < 0:
        raise ValueError("--attenuation must be >= 0")

    configure_calibrator_limits(args)

    print("=== Detecting USRP...")
    usrp = uhd.usrp.MultiUSRP(args.args)

    print(f"=== Measurement direction: {args.dir}")
    print("=== Initializing measurement device...")

    meas_dev = uhd.usrp.cal.get_meas_device(
        args.dir,
        args.meas_dev,
        args.meas_option,
    )

    # Same principle as official uhd_power_cal.py:
    # P_generator = P_DUT + attenuation
    meas_dev.power_offset = args.attenuation

    if args.dir == "tx":
        meas_dev.power_offset -= (
            20.0 * math.log10(args.amplitude)
        )

    print(
        f"=== Measurement path attenuation: "
        f"{args.attenuation:.2f} dB"
    )

    switch = uhd.usrp.cal.get_switch(
        args.dir,
        args.switch,
        args.switch_option,
    )

    usrp_cal = uhd.usrp.cal.get_usrp_calibrator(
        usrp,
        meas_dev,
        args.dir,
        gain_step=args.gain_step,
    )

    if args.dir == "rx" and args.min_input_power is not None:
        usrp_cal.min_detectable_signal = float(args.min_input_power)

        print(
            f"=== RX calibration starting input power: "
            f"{usrp_cal.min_detectable_signal:.2f} dBm"
        )

    channels, antennas, rate = sanitize_args(
        usrp,
        args,
        usrp_cal.default_rate,
    )

    configure_explicit_gains(
        usrp,
        usrp_cal,
        args,
        channels[0],
    )

    results = init_results(args.load)

    usrp_cal.init(
        rate=rate,
        tone_freq=args.tone_freq,
        amplitude=args.amplitude,
    )

    runner = CalRunner(
        usrp,
        usrp_cal,
        meas_dev,
        args,
    )

    try:
        for channel in channels:
            results.setdefault(channel, {})

            for antenna in antennas:
                if antenna in results[channel]:
                    continue

                getattr(
                    usrp,
                    f"set_{args.dir}_antenna",
                )(
                    antenna,
                    channel,
                )

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
                        runner.run(channel, frequency)
                except Exception:
                    usrp_cal.stop(store=False)
                    raise

                results[channel][antenna] = usrp_cal.results
                usrp_cal.stop()

    finally:
        try:
            meas_dev.enable(False)
        except Exception:
            pass

        if hasattr(meas_dev, "close"):
            try:
                meas_dev.close()
            except Exception:
                pass

    if args.store:
        output = Path(args.store).expanduser()

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open("wb") as file:
            pickle.dump(results, file)

        print(f"=== Calibration stored: {output}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nERROR: Calibration interrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
