import numpy as np
from numpy import pi, sqrt, cos, log10, abs, conj, max, ceil, sum, sin
from array_utils import create_empty_array, find_close_index, place
SPEED_LIGHT = 3e8


def get_sampling_interval(sampling_frequency):
    return 1.0 / sampling_frequency


def get_bandwidth(starting_frequency, frequency_at_pulse_time):
    return frequency_at_pulse_time - starting_frequency


def get_time_array(samples_signal, sampling_interval):
    return np.arange(samples_signal) * sampling_interval


def get_samples(duration, sampling_frequency):
    return int(duration * sampling_frequency)


def get_time_signal(sampling_frequency, pulse_duration):
    sampling_interval = get_sampling_interval(sampling_frequency)
    samples_signal = get_samples(pulse_duration, sampling_frequency)
    return get_time_array(samples_signal, sampling_interval)


def slew_rate(pulse_bandwidth, pulse_duration):
    return pi * pulse_bandwidth / pulse_duration


def weight_signal(signal):
    return np.array(np.complex_(signal.real * hamming_window(signal)))


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


def generate_reflected_pulse(signal, pulse_duration, recurrence_interval, time_receipt, sampling_frequency):
    reception_time = recurrence_interval - pulse_duration
    time_array = get_time_signal(sampling_frequency, reception_time) + pulse_duration
    # time_array = correct_time(time_array, time_receipt)
    reflected_pulse = create_empty_array(len(time_array), array_type='complex')
    mark = find_close_index(time_array, time_receipt)
    return place(reflected_pulse, signal, mark), time_array


def matched_filtering(weight_lfm, reflected_signal, quantity):
    reference_signal = conj(weight_lfm[::-1])
    reference_fft = np.fft.fft(reference_signal[::-1], quantity)
    reflected_fft = np.fft.fft(reflected_signal[::-1], quantity)
    fft_mul = reference_fft * reflected_fft
    y_comp = np.fft.ifft(fft_mul, quantity)
    y_abs = abs(y_comp)
    y_abs = y_abs[:len(reflected_signal)]
    y_abs /= max(y_abs)
    return (20 * log10(y_abs))[::-1]


def doppler_effect(reflected_signal, wavelength, speed, sampling_frequency,
                   recurrence_interval, quantity, distance_to_target):

    carrier_frequency = SPEED_LIGHT / wavelength
    doppler_frequency = 2 * speed * carrier_frequency / SPEED_LIGHT
    sampling_interval = get_sampling_interval(sampling_frequency)

    output = create_empty_array(quantity, array_type='object')
    distances = create_empty_array(quantity)

    for k, _ in enumerate(output):
        temp = create_empty_array(len(reflected_signal), array_type='complex')
        for n, _ in enumerate(reflected_signal):
            phi = 2 * pi * doppler_frequency * (n * sampling_interval + k * recurrence_interval)
            temp[n] = complex(cos(phi), sin(phi)) * reflected_signal[n]
        output[k] = temp
        distances[k] = distance_to_target + speed * k * recurrence_interval
    return output, distances


# def get_distance_array(signal, sampling_frequency, time_receipt):
#     sampling_interval = get_sampling_interval(sampling_frequency)
#     samples_signal = get_samples(time_receipt, sampling_frequency)
#     samples_to_target = samples_signal + len(signal)
#     time_array = get_time_array(samples_to_target, sampling_interval)
#     time_array, error = correct_time(time_array, time_receipt)
#     return time_array[samples_signal:] * SPEED_LIGHT / 2, error


def get_distance_array(time_array):
    return time_array * SPEED_LIGHT / 2


def correct_time(time_array, time_receipt):
    index = find_close_index(time_array, time_receipt)
    if time_array[index] == time_receipt:
        return time_array
    error = time_receipt - time_array[index - 1]
    return time_array + error


def get_time(distance):
    return distance * 2 / 3e8


def get_distance(time):
    return SPEED_LIGHT * time / 2


def compress_signals(weight_lfm, reflected_pulses):
    quantity = len(reflected_pulses)
    signals_compression = []
    for signal in reflected_pulses:
        compressed_signal = matched_filtering(weight_lfm, signal, quantity * 2)
        signals_compression.append(compressed_signal)
    return signals_compression


def amplitude_accumulation(compressed_signals):
    size = len(compressed_signals[0])
    output = create_empty_array(size, array_type='complex')

    for k, _ in enumerate(output):
        for signal in compressed_signals:
            output[k] += signal[k].real
    return output
