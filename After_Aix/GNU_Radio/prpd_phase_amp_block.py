import numpy as np
from gnuradio import gr
from scipy.signal import butter, sosfilt, find_peaks


class blk(gr.basic_block):
    def __init__(
        self,
        samp_rate=30e6,
        f_offset=5e6,
        t_start=1.0,
        f_ref=50.0,
        cutoff_if=1e6,
        threshold_factor=4.0,
        min_distance_us=200.0,
        phase_offset_deg=0.0
    ):
        gr.basic_block.__init__(
            self,
            name="PRPD Phase Amplitude Block",
            in_sig=[np.complex64],
            out_sig=[np.float32, np.float32]
        )

        self.samp_rate = float(samp_rate)
        self.f_offset = float(f_offset)
        self.t_start = float(t_start)
        self.f_ref = float(f_ref)
        self.cutoff_if = float(cutoff_if)
        self.threshold_factor = float(threshold_factor)
        self.phase_offset_deg = float(phase_offset_deg)

        self.min_distance = int(min_distance_us * 1e-6 * self.samp_rate)
        if self.min_distance < 1:
            self.min_distance = 1

        self.sample_index = 0

        self.sos_if = butter(
            4,
            self.cutoff_if / (self.samp_rate / 2),
            btype="low",
            output="sos"
        )

        self.zi_if = np.zeros(
            (self.sos_if.shape[0], 2),
            dtype=np.complex64
        )

        self.pending_phases = np.array([], dtype=np.float32)
        self.pending_amps = np.array([], dtype=np.float32)

        print("PRPD Phase Amplitude Block started")
        print("samp_rate:", self.samp_rate)
        print("f_offset:", self.f_offset)
        print("t_start:", self.t_start)
        print("f_ref:", self.f_ref)

    def general_work(self, input_items, output_items):
        out_phase = output_items[0]
        out_amp = output_items[1]

        max_out = len(out_phase)
        produced = 0

        # =========================================================
        # 1) D'abord sortir les anciens points en attente
        # =========================================================
        if len(self.pending_phases) > 0:
            n = min(max_out, len(self.pending_phases))

            out_phase[:n] = self.pending_phases[:n]
            out_amp[:n] = self.pending_amps[:n]

            self.pending_phases = self.pending_phases[n:]
            self.pending_amps = self.pending_amps[n:]

            produced += n

            if produced == max_out:
                return produced

        # =========================================================
        # 2) Lire les nouveaux samples IQ
        # =========================================================
        samples = np.asarray(input_items[0], dtype=np.complex64)
        N = len(samples)

        if N == 0:
            return produced

        # Temps absolu pour phase PRPD
        n_abs = self.sample_index + np.arange(N)
        t_abs = self.t_start + n_abs / self.samp_rate

        # Temps continu pour translation fréquentielle
        t_mix = n_abs / self.samp_rate

        # Suppression DC
        x = samples - np.mean(samples)

        # Translation vers DC
        x = x * np.exp(
            -1j * 2 * np.pi * self.f_offset * t_mix
        )

        # Filtrage IF avec mémoire entre les appels
        x_filt, self.zi_if = sosfilt(
            self.sos_if,
            x,
            zi=self.zi_if
        )

        # Enveloppe
        envelope = np.abs(x_filt)

        # Seuil
        noise_level = np.median(envelope)
        noise_std = np.std(envelope)
        threshold = noise_level + self.threshold_factor * noise_std

        # Détection des pics
        peaks, props = find_peaks(
            envelope,
            height=threshold,
            distance=self.min_distance
        )

        phases = np.array([], dtype=np.float32)
        amps_dbfs = np.array([], dtype=np.float32)

        if len(peaks) > 0:
            t_peaks = t_abs[peaks]

            period = 1.0 / self.f_ref

            phases = (
                ((t_peaks % period) / period) * 360.0
                + self.phase_offset_deg
            ) % 360.0

            amps = envelope[peaks]

            amps_dbfs = 20 * np.log10(
                np.clip(amps, 1e-12, None)
            )

            phases = phases.astype(np.float32)
            amps_dbfs = amps_dbfs.astype(np.float32)

        # On consomme tous les samples d'entrée
        self.consume_each(N)
        self.sample_index += N

        # =========================================================
        # 3) Envoyer phase/amplitude vers les sorties
        # =========================================================
        if len(phases) > 0:
            remaining_space = max_out - produced
            n_send = min(remaining_space, len(phases))

            if n_send > 0:
                out_phase[produced:produced + n_send] = phases[:n_send]
                out_amp[produced:produced + n_send] = amps_dbfs[:n_send]
                produced += n_send

            # Garder les points non envoyés pour le prochain appel
            if n_send < len(phases):
                self.pending_phases = phases[n_send:]
                self.pending_amps = amps_dbfs[n_send:]

        return produced
