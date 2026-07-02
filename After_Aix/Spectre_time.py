import numpy as np
import matplotlib.pyplot as plt
import uhd
import time
from scipy.signal import butter, sosfiltfilt
import time



F_START = 90e6    
F_STOP  = 2e9    

RATE = 12e6           
GAIN = 40
CHANNEL = 0
ANTENNA = "RX2"

DISCARD_TIME = 0.002
USEFUL_DURATION = 0.02
ACQ_DURATION = USEFUL_DURATION + DISCARD_TIME
STEP_HZ =12e6        


LP_CUTOFF_HZ = 4e6    

ROBUST_PERCENTILE = 99.99
REMOVE_DC = False




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




def temporal_band_metric(samples, rate):
    samples = np.asarray(samples, dtype=np.complex64).ravel()

    if len(samples) == 0:
        return np.nan, np.nan

    if REMOVE_DC:
        samples = samples - np.mean(samples)

    discard = int(DISCARD_TIME * rate)  # delet 2 ms
    if len(samples) > discard:
        samples = samples[discard:]
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

    med_amp = np.median(envelope)

    # Conversion dBFS approximative
    max_dbfs = 20 * np.log10(np.clip(max_amp, 1e-12, None))
    robust_dbfs = 20 * np.log10(np.clip(robust_max, 1e-12, None))
    med_dbfs = 20 * np.log10(np.clip(med_amp, 1e-12, None))

    return max_dbfs, robust_dbfs, med_dbfs




def scan_pd_time_domain():
    print("Initialisation USRP...")
    usrp = uhd.usrp.MultiUSRP()

    freqs = np.arange(F_START, F_STOP + STEP_HZ, STEP_HZ)

    max_values = []
    robust_values = []
    med_values = []
    # Gain_l = 40
    print("freqs = ", freqs)
    for i, freq in enumerate(freqs):
        print(f"[{i+1}/{len(freqs)}] Acquisition à {freq/1e6:.1f} MHz")
        print("Freq central = ", freq)

        samples = acquire_time_domain(
            usrp=usrp,
            freq_center=freq,
            rate=RATE,
            duration=ACQ_DURATION,
            gain=GAIN,
            antenna=ANTENNA
        )
        # Time_domain_gr(samples, RATE)
    
        max_dbfs, robust_dbfs, med_dbfs= temporal_band_metric(samples, RATE)

        max_values.append(max_dbfs)
        robust_values.append(robust_dbfs)
        med_values.append(med_dbfs)
        # time.sleep(0.1)

        print(f"   Max = {max_dbfs:.2f} dBFS | Robust = {robust_dbfs:.2f} dBFS| Median = {med_dbfs:.2f} dBFS")

    return freqs, np.array(max_values), np.array(robust_values), np.array(med_values)




def plot_temporal_scan(freqs, max_values, robust_values, med_values):
    print("len(max_values) = ", len(max_values))
    plt.figure(figsize=(13, 5))
    a = -53.16
    # a = np.argmax(max_values.max())
    # print("a = ", a)
    plt.plot(freqs / 1e6, max_values, label="Max temporel")
    print("Max = ", max_values.max())
    # plt.scatter(a / 1e6, max_values.max(), label="Max temporel")
    
    plt.plot(freqs / 1e6, robust_values, label=f"Percentile {ROBUST_PERCENTILE}%")
    # # plt.plot(freqs / 1e6, med_values, label=f"Percentile {50}%")
    plt.axhline(a, color='r', linestyle='--', label=f"Mean (None PD Gain = {GAIN} db) = {a:.2f}")
    plt.axhline(max_values.mean(), color='green', linestyle='--', label=f"Mean (Current PD Gain = {GAIN} db) = {max_values.mean():.2f}")

    plt.title("Détection de pulses par scan temporel")
    plt.xlabel("Fréquence centrale RX (MHz)")
    plt.ylabel("Amplitude max détectée (dBFS)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def Time_domain_gr(samples, rate, name = ""):
    t = np.arange(0, ACQ_DURATION, 1/rate)
    signal_50 = 0.1 * np.sin(2*np.pi*50*t)
    alpha = int(len(t)/40)
    plt.figure(figsize=(12, 5))
    plt.plot(t[:], np.real(samples)[:], label = "Real part")
    plt.plot(t[:], np.imag(samples)[:], label = "Imag part")
    # plt.plot(t[:alpha], signal_50[:alpha], label = "Signal 50 Hz")
    plt.title("Time Sink")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.legend()
    plt.grid()
    plt.savefig(f"./Main_figs/Time_domain_plot{name}.png")
    plt.show()


def main():
    start = time.perf_counter()
    freqs, max_values, robust_values, med_values= scan_pd_time_domain()
    end = time.perf_counter()
    print("Time duration is : ", np.abs(end - start))
    plot_temporal_scan(freqs, max_values, robust_values, med_values)


if __name__ == "__main__":
    main()