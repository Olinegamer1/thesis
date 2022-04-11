import numpy as np


def get_sample_interval(sampling_frequency):
    return 1.0 / sampling_frequency


def get_bandwidth(starting_frequency, frequency_at_pulse_time):
    return frequency_at_pulse_time - starting_frequency


def get_time_array(samples_signal, sampling_interval):
    return np.arange(samples_signal) * sampling_interval


def get_samples(duration, sampling_frequency):
    return int(duration * sampling_frequency)
