import numpy as np

from functools import reduce
from numpy import pi, sqrt, cos
from array_utils import create_empty_array


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


def weight_signal(signal, hamming_win):
    output = []
    for index, _ in enumerate(signal):
        output.append(reduce(lambda x, y: (x * hamming_win[index], y * hamming_win[index]), signal[index]))
    return output


def hamming_window(signal):
    size = len(signal)
    output = create_empty_array(size)

    for k, _ in enumerate(signal):
        if k < size / 2:
            output[k] = 0.54 - 0.46 * cos(2 * pi * (size - k - 1) / (size - 1))
        output[k] = 0.54 - 0.46 * cos(2 * pi * k / (size - 1))

    return output


def norm(signal):
    output = 0
    for k, _ in enumerate(signal):
        output += signal[k] ** 2
    return sqrt(output)


def signal_multiplier(db):
    return 10 ** (db / 20)


def noise_signal_normalization(signal_norm, noise_norm, noise, signal_mul):
    size = len(noise)
    output = create_empty_array(size)

    for k, _ in enumerate(output):
        output[k] = signal_norm * noise[k] / (signal_mul * noise_norm)
    return output


def signal_scale(signal, noise_norm, signal_norm, signal_mul):
    size = len(signal)
    output = create_empty_array(size)

    for k, _ in enumerate(output):
        output[k] = noise_norm * signal_mul * signal[k] / signal_norm
    return output


def additive_signal_mixture(signal, noise):
    size = len(signal)
    output = create_empty_array(size)

    for k, _ in enumerate(output):
        output[k] = signal[k] + noise[k]

    return output


def add_zeroes(pulse, count):
    size = len(pulse)
    output = create_empty_array(size + count)
    i = 0
    for k, _ in enumerate(output):
        if count / 2 < k < len(pulse) + count / 2:
            output[k] = pulse[i]
            i += 1
    return output
