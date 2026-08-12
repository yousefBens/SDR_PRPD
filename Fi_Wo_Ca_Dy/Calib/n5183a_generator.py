#!/usr/bin/env python3
import pyvisa
from uhd.usrp.cal.meas_device import SignalGeneratorBase

class N5183APowerGenerator(SignalGeneratorBase):
    key = "n5183a"

    def __init__(self, options):
        super().__init__(options)

        self.ip = options.get("ip", "192.168.1.100")
        backend = options.get("backend", "@py")

        self.min_generator_dbm = float(options.get("min_generator_dbm", -20.0))
        self.max_generator_dbm = float(options.get("max_generator_dbm", 20.0))
        self.max_output_power = float(options.get("max_dut_input_dbm", -20.0))

        forced_resource = options.get("resource")
        resources = (
            [forced_resource]
            if forced_resource
            else [
                f"TCPIP0::{self.ip}::inst0::INSTR",
                f"TCPIP0::{self.ip}::5025::SOCKET",
            ]
        )

        self.rm = pyvisa.ResourceManager(backend)
        self.dev = None
        last_error = None

        for resource in resources:
            print(f"[N5183A] Trying resource: {resource}")
            dev = None
            try:
                dev = self.rm.open_resource(resource, open_timeout=5000)
                dev.timeout = 10000
                dev.write_termination = "\n"
                dev.read_termination = "\n"

                identity = dev.query("*IDN?").strip()
                print(f"[N5183A] Connected: {identity}")

                if "N5183A" not in identity.upper():
                    raise RuntimeError(f"Instrument is not an N5183A: {identity}")

                self.dev = dev
                self.resource = resource
                break
            except Exception as exc:
                last_error = exc
                print(f"[N5183A] Connection failed: {exc}")
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
                f"Unable to connect to N5183A at {self.ip}. Last error: {last_error}"
            )

        print(f"[N5183A] Active resource: {self.resource}")
        print(
            f"[N5183A] Generator limits: "
            f"{self.min_generator_dbm:.1f} -> {self.max_generator_dbm:.1f} dBm"
        )
        print(
            f"[N5183A] RX2/DUT maximum allowed: "
            f"{self.max_output_power:.1f} dBm"
        )

        self.dev.write(":OUTP OFF")
        self.dev.write(":FREQ:MODE FIX")
        self.dev.write(":POW:MODE FIX")
        self._check_errors("initialization")

    def enable(self, enable=True):
        self.dev.write(":OUTP ON" if enable else ":OUTP OFF")
        state = int(float(self.dev.query(":OUTP?")))
        if state != int(enable):
            raise RuntimeError("Unable to change RF output state")
        print("[N5183A] RF output:", "ON" if state else "OFF")
        self._check_errors("RF output")

    def set_frequency(self, freq):
        freq = float(freq)
        self.dev.write(f":FREQ {freq:.3f} HZ")
        actual = float(self.dev.query(":FREQ?"))
        if abs(actual - freq) > 1.0:
            raise RuntimeError(
                f"Frequency mismatch: requested {freq:.3f}, actual {actual:.3f}"
            )
        print(f"[N5183A] Frequency = {actual/1e6:.6f} MHz")
        self._check_errors("frequency")

    def _set_power(self, generator_power_dbm):
        generator_power_dbm = float(generator_power_dbm)

        if generator_power_dbm < self.min_generator_dbm:
            raise RuntimeError(
                f"N5183A requested {generator_power_dbm:.2f} dBm, "
                f"below minimum {self.min_generator_dbm:.2f} dBm"
            )

        if generator_power_dbm > self.max_generator_dbm:
            raise RuntimeError(
                f"N5183A requested {generator_power_dbm:.2f} dBm, "
                f"above maximum {self.max_generator_dbm:.2f} dBm"
            )

        self.dev.write(f":POW {generator_power_dbm:.3f} DBM")
        actual = self._get_power()

        if abs(actual - generator_power_dbm) > 0.10:
            self.enable(False)
            raise RuntimeError(
                f"N5183A power mismatch: requested "
                f"{generator_power_dbm:.3f}, actual {actual:.3f} dBm"
            )

        print(f"[N5183A] Generator power = {actual:.3f} dBm")
        self._check_errors("power")
        return actual

    def _get_power(self):
        return float(self.dev.query(":POW?"))

    def _check_errors(self, operation):
        errors = []
        while True:
            response = self.dev.query("SYST:ERR?").strip()
            if response.startswith("+0") or response.startswith("0"):
                break
            errors.append(response)

        if errors:
            raise RuntimeError(
                f"N5183A SCPI error during {operation}: " + " | ".join(errors)
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
