import pyvisa

from uhd.usrp.cal.meas_device import SignalGeneratorBase


class N5183APowerGenerator(SignalGeneratorBase):
    """
    Pilote Agilent N5183A pour uhd_power_cal.py.

    Le paramètre --attenuation est traité par UHD :
        puissance générateur =
            puissance demandée à l'entrée du B200 + atténuation
    """

    key = "n5183a"

    def __init__(self, options):
        super().__init__(options)

        ip = options.get("ip", "192.168.1.100")
        backend = options.get("backend", "@py")

        # Limites physiques du générateur.
        # D'après ton banc, -20 dBm est le niveau le plus faible.
        self.min_generator_dbm = float(
            options.get("min_generator_dbm", -20.0)
        )

        # Mets ici la puissance maximale réellement autorisée
        # sur ton générateur. On commence prudemment à +10 dBm.
        self.max_generator_dbm = float(
            options.get("max_generator_dbm", 10.0)
        )

        # Limite de sécurité au connecteur RX2 du B200.
        # UHD compare ses demandes à cette limite avant
        # d'ajouter l'atténuation.
        self.max_output_power = float(
            options.get("max_dut_input_dbm", -15.0)
        )

        resource = options.get(
            "resource",
            f"TCPIP0::{ip}::inst0::INSTR",
        )

        print(f"[N5183A] Connexion : {resource}")

        self.rm = pyvisa.ResourceManager(backend)
        self.dev = self.rm.open_resource(resource)

        self.dev.timeout = 10000
        self.dev.write_termination = "\n"
        self.dev.read_termination = "\n"

        identity = self.dev.query("*IDN?").strip()
        print(f"[N5183A] Identité : {identity}")

        if "N5183A" not in identity.upper():
            raise RuntimeError(
                f"Le périphérique n'est pas un N5183A : {identity}"
            )

        # Mise en sécurité au démarrage
        self.dev.write(":OUTP OFF")
        self.dev.write(":FREQ:MODE FIX")
        self.dev.write(":POW:MODE FIX")

        self._check_errors("initialisation")

    def enable(self, enable=True):
        self.dev.write(":OUTP ON" if enable else ":OUTP OFF")

        state = int(float(self.dev.query(":OUTP?")))

        print(
            "[N5183A] RF Output :",
            "ON" if state else "OFF"
        )

        self._check_errors("commande RF")

    def set_frequency(self, freq):
        freq = float(freq)

        self.dev.write(f":FREQ {freq:.3f} HZ")

        actual_freq = float(
            self.dev.query(":FREQ?")
        )

        print(
            f"[N5183A] Fréquence = "
            f"{actual_freq / 1e6:.3f} MHz"
        )

        self._check_errors("réglage fréquence")

    def _set_power(self, generator_power_dbm):
        """
        Cette fonction reçoit la puissance à configurer
        réellement sur le N5183A.

        UHD a déjà ajouté les 60 dB d'atténuation.
        """

        generator_power_dbm = float(generator_power_dbm)

        if generator_power_dbm < self.min_generator_dbm:
            raise RuntimeError(
                f"N5183A : {generator_power_dbm:.2f} dBm demandé, "
                f"mais le minimum disponible est "
                f"{self.min_generator_dbm:.2f} dBm."
            )

        if generator_power_dbm > self.max_generator_dbm:
            raise RuntimeError(
                f"N5183A : {generator_power_dbm:.2f} dBm demandé, "
                f"mais la limite configurée est "
                f"{self.max_generator_dbm:.2f} dBm."
            )

        self.dev.write(
            f":POW {generator_power_dbm:.3f} DBM"
        )

        actual_power = self._get_power()

        print(
            f"[N5183A] Puissance générateur = "
            f"{actual_power:.2f} dBm"
        )

        self._check_errors("réglage puissance")

        return actual_power

    def _get_power(self):
        return float(
            self.dev.query(":POW?")
        )

    def _check_errors(self, operation):
        errors = []

        while True:
            response = self.dev.query("SYST:ERR?").strip()

            if (
                response.startswith("+0")
                or response.startswith("0")
            ):
                break

            errors.append(response)

        if errors:
            raise RuntimeError(
                f"Erreur SCPI pendant {operation} : "
                + " | ".join(errors)
            )

    def close(self):
        try:
            self.dev.write(":OUTP OFF")
        finally:
            self.dev.close()
            self.rm.close()
