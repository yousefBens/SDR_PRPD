import numpy as np
import matplotlib.pyplot as plt
import uhd
import time
from scipy.signal import butter, sosfiltfilt

# =====================================================
# CONFIGURATION
# =====================================================

F_START = 100e6       # début scan : 100 MHz
F_STOP  = 2e9         # fin scan : 2 GHz

RATE = 12e6           # fréquence d'échantillonnage B200
GAIN = 40
CHANNEL = 0
ANTENNA = "RX2"

ACQ_DURATION = 0.02   # durée acquisition par petite bande en secondes
STEP_HZ = 5e6         # pas entre deux fréquences centrales


LP_CUTOFF_HZ = 5e6    # largeur utile après filtrage

ROBUST_PERCENTILE = 99.9
REMOVE_DC = True


# =====================================================
# ACQUISITION USRP
# =====================================================

def acquire_time_domain(usrp, freq_center, rate, duration, gain, antenna):
    num_samps = int(rate * duration)

    usrp.set_rx_rate(rate, CHANNEL)
    usrp.set_rx_freq(uhd.types.TuneRequest(freq_center), CHANNEL)
    usrp.set_rx_gain(gain, CHANNEL)
    usrp.set_rx_antenna(antenna, CHANNEL)

    usrp.set_rx_dc_offset(True, CHANNEL)
    usrp.set_rx_iq_balance(True, CHANNEL)

    time.sleep(0.05)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [CHANNEL]

    rx_streamer = usrp.get_rx_stream(stream_args)
    rx_md = uhd.types.RXMetadata()

    samples = np.zeros(num_samps, dtype=np.complex64)
    buff = np.zeros((1, 4096), dtype=np.complex64)

    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samps
    stream_cmd.stream_now = True

    rx_streamer.issue_stream_cmd(stream_cmd)

    total = 0

    while total < num_samps:
        n = rx_streamer.recv(buff, rx_md, timeout=3.0)

        if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
            print("RX error:", rx_md.strerror())
            continue

        if n > 0:
            end = min(total + n, num_samps)
            samples[total:end] = buff[0, :end-total]
            total = end

    return samples[:total]


# =====================================================
# TRAITEMENT TEMPOREL POUR UNE BANDE
# =====================================================

def temporal_band_metric(samples, rate):
    samples = np.asarray(samples, dtype=np.complex64).ravel()

    if len(samples) == 0:
        return np.nan, np.nan

    if REMOVE_DC:
        samples = samples - np.mean(samples)

    # Filtre passe-bas pour garder seulement la bande utile
    cutoff = min(LP_CUTOFF_HZ, 0.45 * rate)

    sos = butter(
        4,
        cutoff / (rate / 2),
        btype="low",
        output="sos"
    )

    samples_filt = sosfiltfilt(sos, samples)

    # Enveloppe temporelle
    envelope = np.abs(samples_filt)

    # Max classique
    max_amp = np.max(envelope)

    # Max robuste : évite qu'un seul point parasite domine
    robust_max = np.percentile(envelope, ROBUST_PERCENTILE)

    # Conversion dBFS approximative
    max_dbfs = 20 * np.log10(np.clip(max_amp, 1e-12, None))
    robust_dbfs = 20 * np.log10(np.clip(robust_max, 1e-12, None))

    return max_dbfs, robust_dbfs


# =====================================================
# SCAN FREQUENTIEL
# =====================================================

def scan_pd_time_domain():
    print("Initialisation USRP...")
    usrp = uhd.usrp.MultiUSRP()

    freqs = np.arange(F_START, F_STOP + STEP_HZ, STEP_HZ)

    max_values = []
    robust_values = []
    Gain_l = 40
    for i, freq in enumerate(freqs):
        print(f"[{i+1}/{len(freqs)}] Acquisition à {freq/1e6:.1f} MHz")

        samples = acquire_time_domain(
            usrp=usrp,
            freq_center=freq,
            rate=RATE,
            duration=ACQ_DURATION,
            gain=Gain_l,
            antenna=ANTENNA
        )

        max_dbfs, robust_dbfs = temporal_band_metric(samples, RATE)

        max_values.append(max_dbfs)
        robust_values.append(robust_dbfs)

        print(f"   Max = {max_dbfs:.2f} dBFS | Robust = {robust_dbfs:.2f} dBFS")

    return freqs, np.array(max_values), np.array(robust_values)


# =====================================================
# AFFICHAGE
# =====================================================

def plot_temporal_scan(freqs, max_values, robust_values):
    print("len(max_values) = ", len(max_values))
    plt.figure(figsize=(13, 5))

    plt.plot(freqs / 1e6, max_values, label="Max temporel")
    plt.plot(freqs / 1e6, robust_values, label=f"Percentile {ROBUST_PERCENTILE}%")

    plt.title("Détection de pulses par scan temporel")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Amplitude max détectée (dBFS)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =====================================================
# MAIN
# =====================================================

def main():
    freqs, max_values, robust_values = scan_pd_time_domain()

    # np.savez(
    #     "scan_pd_time_domain_100MHz_2GHz.npz",
    #     freqs=freqs,
    #     max_dbfs=max_values,
    #     robust_dbfs=robust_values
    # )

    plot_temporal_scan(freqs, max_values, robust_values)


if __name__ == "__main__":
    main()