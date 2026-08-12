FINAL STRUCTURE

Calib/
├── calibrate.py
├── uhd_power_cal_adapted.py
├── n5183a_generator.py
└── Cal_Files/

RUN:
    python3 calibrate.py

Only edit the CONFIGURATION section at the top of calibrate.py.

The measurement engine still uses UHD:
    get_meas_device()
    get_usrp_calibrator()
    init_frequencies()
    run_rx_cal()
    usrp_cal.results

Hardware adaptation:
    ATT 60 dB -> gains 60 / 76
    ATT 40 dB -> gains 20 / 40
    ATT 20 dB -> gain 0
    Generator physical limits: -20..+20 dBm
    RX2 safe max: -20 dBm
    RX2 dangerous level: -15 dBm
    Gain max: 76 dB
