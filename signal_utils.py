import numpy as np

from numpy import pi


def get_sampling_interval(sampling_frequency):
    return 1.0 / sampling_frequency


def get_bandwidth(starting_frequency, frequency_at_pulse_time):
    return frequency_at_pulse_time - starting_frequency


def get_time_array(samples_signal, sampling_interval):
    return np.arange(samples_signal) * sampling_interval * 1000


def get_samples(duration, sampling_frequency):
    return int(duration * sampling_frequency)


def slew_rate(pulse_bandwidth, pulse_duration):
    return pi * pulse_bandwidth / pulse_duration
