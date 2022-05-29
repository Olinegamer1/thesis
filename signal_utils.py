import numpy as np
from numpy import pi, cos, abs, max, ceil, sin
from numpy.linalg import norm

from array_utils import create_empty_array, find_close_index, place, slice_range, find_close_index_ampl, log_scale, \
    generate_noise_array
from plot import show_plot, show_multiplot

SPEED_LIGHT = 3e8


def get_sampling_interval(sampling_frequency):
    return 1.0 / sampling_frequency


def get_bandwidth(starting_frequency, frequency_at_pulse_time):
    return frequency_at_pulse_time - starting_frequency


def get_time_array(samples_signal, sampling_interval):
    return np.arange(samples_signal) * sampling_interval


def get_samples(duration, sampling_frequency):
    return ceil(duration * sampling_frequency)


def get_time_signal(sampling_frequency, pulse_duration):
    sampling_interval = get_sampling_interval(sampling_frequency)
    samples_signal = get_samples(pulse_duration, sampling_frequency)
    return get_time_array(samples_signal, sampling_interval)


def slew_rate(pulse_bandwidth, pulse_duration):
    return pi * pulse_bandwidth / pulse_duration


def weight_signal_hamming(signal):
    size = len(signal)
    return np.array(np.complex_(signal.real * np.hamming(size)))


def signal_multiplier(ratio):
    return 10 ** (ratio / 20)


def noise_signal_normalization(signal_norm, noise_norm, noise, signal_mul):
    factor = signal_norm / (signal_mul * noise_norm)
    return noise * factor


def signal_scale(signal, noise_norm, signal_norm, signal_mul):
    factor = (noise_norm * signal_mul) / signal_norm
    return signal * factor


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
    requested_signal = signal_scale(signal, noise_norm, signal_norm, signal_multiplier(ratio))
    return requested_signal + noise


def generate_reflected_pulse(signal, recurrence_interval, time_receipt, sampling_frequency):
    sampling_interval = get_sampling_interval(sampling_frequency)
    samples_signal = get_samples(recurrence_interval, sampling_frequency)
    time_array = get_time_array(samples_signal, sampling_interval)
    reflected_pulse = create_empty_array(len(time_array), array_type='complex_')
    mark = find_close_index(time_array, time_receipt)
    return place(reflected_pulse, signal, mark)[len(signal):], time_array[len(signal):]


def matched_filtering(weight_lfm, reflected_signal):
    size = len(weight_lfm) + len(reflected_signal)
    reference_signal = weight_lfm.conjugate()
    reference_fft = np.fft.fft(reference_signal, size)
    reflected_fft = np.fft.fft(reflected_signal[::-1], size)
    fft_mul = reference_fft * reflected_fft
    y_comp = np.fft.ifft(fft_mul, size)
    y_comp = y_comp[:len(reflected_signal)]
    return y_comp[::-1]


def doppler_effect(reflected_signal, wavelength, speed, sampling_frequency, recurrence_interval, quantity, ratio):
    doppler_frequency = 2 * speed / wavelength
    sampling_interval = get_sampling_interval(sampling_frequency)

    output = []
    for k in range(quantity):
        temp = create_empty_array(len(reflected_signal), array_type='complex')
        for n, _ in enumerate(reflected_signal):
            phi = 2 * pi * doppler_frequency * (n * sampling_interval + (k + 1) * recurrence_interval)
            temp[n] = complex(cos(phi), sin(phi)) * reflected_signal[n]
        noise = generate_noise_array(1000)
        noisy_signal = mix_SNR(temp, noise, ratio)
        output.append(noisy_signal)
    return output


def get_distance_array_reflected_pulses(reflected_pulses, distance_array,
                                        distance_to_target, speed, recurrence_interval):
    output = []
    for k, _ in enumerate(reflected_pulses):
        shift = distance_to_target + speed * k * recurrence_interval
        output.append(correct_distance(distance_array, shift, distance_to_target))
    return output


def get_distance_array_from_time(time_array):
    return time_array * SPEED_LIGHT / 2


def correct_distance(distance_array, distance_new_target, distance_to_target):
    output = np.copy(distance_array)
    distance_new_target -= distance_to_target
    return output + distance_new_target


def get_time(distance):
    return distance * 2 / SPEED_LIGHT


def get_distance(time):
    return SPEED_LIGHT * time / 2


def compress_signals_by_module(weight_lfm, reflected_pulses):
    compressed_by_module = compress_signals(weight_lfm, reflected_pulses)
    return abs(compressed_by_module)


def compress_signals(weight_lfm, reflected_pulses):
    len(reflected_pulses)
    signals_compression = []
    for signal in reflected_pulses:
        compressed_signal = matched_filtering(weight_lfm, signal)
        signals_compression.append(compressed_signal)
    return signals_compression


def amplitude_accumulation(compressed_signals, distance_arrays, distance):
    size = len(compressed_signals[0])
    output = create_empty_array(size, array_type='complex')
    for k, signal in enumerate(compressed_signals):
        max_index = find_close_index(signal, max(signal))
        max_dist_sig = distance_arrays[k][max_index]
        max_new_index = find_close_index_ampl(distance, max_dist_sig)
        shift = max_new_index - max_index
        output += np.roll(signal, shift)
    return output


def interpolation(signal, distance_array, sampling_frequency):
    maximum = find_close_index(signal, max(signal))
    left = signal[maximum - 1]
    mid = signal[maximum]
    right = signal[maximum + 1]
    sampling_interval = get_sampling_interval(sampling_frequency)
    dk = ((left - right) / (left - 2 * mid + right)) * 0.5
    step_distance = get_distance(sampling_interval)
    distance = (distance_array[maximum] + dk * step_distance)
    polynomial_value = (0.5 * right - mid + 0.5 * left) * dk ** 2 + (0.5 * right - 0.5 * left) * dk + mid
    return polynomial_value, distance


def slice_reflected_pulses(compressed_signals, index, recurrence_interval):
    sliced_pulses = slice_range(compressed_signals, index)
    time_array = [recurrence_interval * k for k in range(len(sliced_pulses))]
    return sliced_pulses, time_array


def weight_signal_blackman(signal):
    size = len(signal)
    return signal * np.blackman(size)


def cog_nak(weight_signal, recurrence_interval, wavelength):
    size = len(weight_signal) * 4
    fft = np.fft.fft(weight_signal, size)
    fft = abs(fft)
    fft = np.fft.fftshift(fft)
    freq = np.arange(-size / 2, size / 2) / size
    freq = wavelength * freq / 2
    freq *= 1 / recurrence_interval
    return fft, freq


def get_radial_speed(signal, doppler_frequencies):
    signal_max = max(signal)
    index_max_value = find_close_index(signal, signal_max)
    return doppler_frequencies[index_max_value]


def without_amplitude(compressed_signals, recurrence_interval, wavelength):
    slices = get_slices(compressed_signals)
    weight_signals = [weight_signal_blackman(slice) for slice in slices]
    cog_naks = []
    for signal in weight_signals:
        cog_naks.append((np.fft.fftshift(np.fft.fft(signal, 2048)), np.arange(-2048 / 2, 2048 / 2) / 2048 * wavelength
                         / 2 * (1 / recurrence_interval)))
    # show_multiplot([freq[1] for freq in cog_naks], [sig[0].real for sig in cog_naks], "БПФ все срезов")

    max_index_slice = 0
    max_slice = 0
    for k, signal in enumerate(cog_naks):
        if max(signal[0]) > max_slice:
            max_slice, max_index_slice = max(signal), k
    speed = get_radial_speed(cog_naks[max_index_slice][0], cog_naks[1][1])
    return cog_naks, speed, max_index_slice


def get_slices(compressed_signals):
    output = []
    for k, _, in enumerate(compressed_signals[0]):
        output.append(slice_range(compressed_signals, k))
    return output
