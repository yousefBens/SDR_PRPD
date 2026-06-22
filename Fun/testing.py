import numpy as np
import matplotlib.pyplot as plt
import uhd

TX_FREQ = 200e6
RATE = 12e6
TX_GAIN = 76
CHANNEL = 0
ANTENNA = "TX/RX"

BUFFER_DURATION = 1.0
PD_OFFSET = 1e6          # donc signal utile à 123 MHz
F_REF = 50.0

PULSE_DURATION = 50e-6
TAU = 40e-6
MIN_GAP_US = 200

TX_LEVEL = 0.8
PILOT_LEVEL = 0.15      # mets 0.15 pour voir le pic RTL, puis 0 pour PRPD pur


def gaussian_phase(phase_deg, center, sigma):
    return np.exp(-((phase_deg - center) ** 2) / (2 * sigma ** 2))


def simulate_pd_signal(num_samps, rate, f_offset):
    t = np.arange(num_samps) / rate
    x = np.zeros(num_samps, dtype=np.complex64)

    phase_ref = (360.0 * F_REF * t) % 360.0

    prob_pd = (
        gaussian_phase(phase_ref, 60, 20)
        + gaussian_phase(phase_ref, 220, 20)
    )
    prob_pd /= np.max(prob_pd)

    rng = np.random.default_rng()
    p_global = 0.08 * prob_pd

    candidates = np.where(rng.random(num_samps) < p_global)[0]

    min_gap_samples = int(MIN_GAP_US * 1e-6 * rate)
    event_indices = []
    last_idx = -min_gap_samples

    for c in candidates:
        if c - last_idx >= min_gap_samples:
            event_indices.append(c)
            last_idx = c

    pulse_len = int(PULSE_DURATION * rate)
    tp = np.arange(pulse_len) / rate

    for idx in event_indices:
        A = rng.uniform(0.5, 1.5)

        pulse_env = A * np.exp(-tp / TAU)
        pulse_carrier = np.exp(1j * 2 * np.pi * f_offset * tp)

        pulse = pulse_env * pulse_carrier

        end_idx = min(idx + pulse_len, num_samps)
        x[idx:end_idx] += pulse[:end_idx - idx]

    print("Nombre de décharges simulées :", len(event_indices))

    # pilot tone continu pour vérifier au RTL-SDR
    if PILOT_LEVEL > 0:
        pilot = PILOT_LEVEL * np.exp(1j * 2 * np.pi * f_offset * t)
        x += pilot.astype(np.complex64)

    return x.astype(np.complex64)


def normalize_signal(x, level=0.8):
    m = np.max(np.abs(x))
    if m > 0:
        x = level * x / m
    return x.astype(np.complex64)


def show_tx_debug(x, rate):
    print("\n========== TX DEBUG ==========")
    print("Amplitude max     :", np.max(np.abs(x)))
    print("Amplitude moyenne :", np.mean(np.abs(x)))
    print("RF utile          :", (TX_FREQ + PD_OFFSET) / 1e6, "MHz")

    n = min(len(x), int(0.02 * rate))
    t = np.arange(n) / rate * 1e3

    fig, axs = plt.subplots(3, 1, figsize=(14, 9))

    axs[0].plot(t, np.real(x[:n]), label="I")
    axs[0].plot(t, np.imag(x[:n]), label="Q")
    axs[0].set_title("Signal TX I/Q")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, np.abs(x[:n]))
    axs[1].set_title("Enveloppe TX")
    axs[1].grid(True)

    NFFT = 8192
    X = np.fft.fftshift(np.fft.fft(x[:NFFT] * np.hanning(NFFT)))
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1 / rate))
    PSD = 20 * np.log10(np.abs(X) + 1e-12)

    axs[2].plot(freqs / 1e6, PSD)
    axs[2].axvline(PD_OFFSET / 1e6, linestyle="--", label="PD_OFFSET")
    axs[2].set_title("Spectre TX baseband")
    axs[2].set_xlabel("Fréquence baseband MHz")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def transmit_forever():
    usrp = uhd.usrp.MultiUSRP()

    usrp.set_tx_rate(RATE, CHANNEL)
    usrp.set_tx_freq(TX_FREQ, CHANNEL)
    usrp.set_tx_gain(TX_GAIN, CHANNEL)
    usrp.set_tx_antenna(ANTENNA, CHANNEL)

    print("\n========== UHD CONFIG ==========")
    print("TX freq réelle :", usrp.get_tx_freq(CHANNEL) / 1e6, "MHz")
    print("TX rate réel  :", usrp.get_tx_rate(CHANNEL) / 1e6, "MS/s")
    print("TX gain réel  :", usrp.get_tx_gain(CHANNEL), "dB")
    print("TX antenna    :", usrp.get_tx_antenna(CHANNEL))
    print("RF utile      :", (TX_FREQ + PD_OFFSET) / 1e6, "MHz")
    print("================================")

    num_samps = int(BUFFER_DURATION * RATE)

    tx_signal = simulate_pd_signal(num_samps, RATE, PD_OFFSET)
    tx_signal = normalize_signal(tx_signal, TX_LEVEL)

    # show_tx_debug(tx_signal, RATE)

    input("\nEntrée pour démarrer l'émission...")

    tx_streamer = usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))

    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = False
    md.has_time_spec = False

    chunk_size = 4096

    print("\nÉmission continue...")
    print("Ctrl+C pour arrêter.")

    try:
        while True:
            for i in range(0, len(tx_signal), chunk_size):
                tx_streamer.send(tx_signal[i:i + chunk_size], md)
                md.start_of_burst = False

    except KeyboardInterrupt:
        print("\nArrêt demandé.")

    finally:
        md.end_of_burst = True
        tx_streamer.send(np.zeros(1, dtype=np.complex64), md)
        print("TX arrêté.")


if __name__ == "__main__":
    transmit_forever()