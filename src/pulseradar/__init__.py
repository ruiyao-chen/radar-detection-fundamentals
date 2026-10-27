"""pulseradar: pulse-radar design and threshold detection, checked against exact answers.

Part 1 sizes an airborne pulse radar that searches for moving ground vehicles:
radar equation, Earth-curvature geometry, land clutter, Doppler and PRF limits,
scan timing and detection probability, with a trade study over PRF and scan
rate. Part 2 detects echoes in a simulated range-gate x pulse matrix: noise
estimated from receiver-only data, a threshold set for a required false-alarm
probability, M-of-N integration over pulses, SNR and radar cross-section
estimates, and a cell-averaging CFAR for a noise floor that changes with range.

    config      physical constants, the scenario (requirements) and the baseline design
    geometry    slant range, grazing angle and ground range over a 4/3 Earth
    radar_eq    antenna gain, noise power, single-pulse SNR, RCS from a measured SNR
    detection   Pfa from false-alarm time; Albersheim; exact Swerling 0/1 noncoherent Pd; M-of-N
    clutter     constant-gamma land clutter, clutter cell area, target-to-clutter ratio
    doppler     clutter Doppler, minimum detectable velocity, blind speeds, range ambiguity
    design      evaluate a design against the scenario; PRF x scan-rate trade study
    echoes      Part 2: echo matrix simulation, noise estimators, thresholds, CFAR
    plotting    report figures
"""

__version__ = "0.1.0"
