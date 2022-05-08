import numpy as np
from numpy import pi, sqrt, cos, log10, abs, conj, max, ceil, sum

from array_utils import create_empty_array


def get_sampling_interval(sampling_frequency):
    return 1.0 / sampling_frequency


def get_bandwidth(starting_frequency, frequency_at_pulse_time):
    return frequency_at_pulse_time - starting_frequency


def get_time_array(samples_signal, sampling_interval):
    return np.arange(samples_signal) * sampling_interval


def get_samples(duration, sampling_frequency):
    return int(duration * sampling_frequency)


def slew_rate(pulse_bandwidth, pulse_duration):
    return pi * pulse_bandwidth / pulse_duration


def weight_signal(signal):
    return signal * hamming_window(signal)


def hamming_window(signal):
    size = len(signal)
    output = create_empty_array(size)

    for k, _ in enumerate(signal):
        if k < size / 2:
            output[k] = 0.54 - 0.46 * cos(2 * pi * (size - k - 1) / (size - 1))
        output[k] = 0.54 - 0.46 * cos(2 * pi * k / (size - 1))

    return output


def norm(signal):
    return sqrt(sum(signal ** 2))


def signal_multiplier(ratio):
    return 10 ** (ratio / 20)


def noise_signal_normalization(signal_norm, noise_norm, noise, signal_mul):
    factor = signal_norm / (signal_mul * noise_norm)
    return noise * factor


def signal_scale(signal, noise_norm, signal_norm, signal_mul):
    factor = (noise_norm * signal_mul) / signal_norm
    return signal * factor


def additive_signal_mixture(signal, noise):
    return signal + noise


def mix_SNR(signal, noise, ratio):
    number_samples = len(signal)

    if len(noise) >= number_samples:
        noise = noise[:number_samples]
    else:
        num_reps = int(ceil(number_samples / len(noise)))
        temp = noise.repeat(num_reps)
        noise = temp[:number_samples]

    signal_norm = norm(signal)
    noise_norm = norm(noise)

    requested_noise = noise_signal_normalization(signal_norm, noise_norm, noise, signal_multiplier(ratio))
    return additive_signal_mixture(signal, requested_noise), requested_noise


def matched_filtering(reference_signal, reflected_signal, NN):
    reference_signal = conj(reference_signal)
    reference_fft = np.fft.fft(reference_signal, NN)
    reflected_fft = np.fft.fft(reflected_signal, NN)
    fft_mul = reference_fft * reflected_fft
    y_comp = np.fft.ifft(fft_mul, NN)
    y_abs = abs(y_comp)
    y_abs = y_abs / max(y_abs)
    return 20 * log10(y_abs)


def get_distance_array(signal, Fs):
    samples = len(signal)
    speed_light = 3e8
    time_array = get_time_array(samples, get_sampling_interval(Fs))
    return time_array * speed_light / 2
