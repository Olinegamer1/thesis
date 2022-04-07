from math import cos, pi, sin
import matplotlib.pyplot as plt
import numpy as np


def get_sample_interval(sampling_frequency):
    return 1.0 / sampling_frequency


def create_empty_array(size, array_type='int64'):
    return np.zeros(size, dtype=array_type)


def get_bandwidth(starting_frequency, frequency_at_pulse_time):
    return frequency_at_pulse_time - starting_frequency


def get_kdt(k, sampling_interval):
    return k * sampling_interval


def get_time_array(samples_signal, sampling_interval):
    return np.arange(samples_signal) * sampling_interval


def get_samples(duration, sampling_frequency):
    return int(duration * sampling_frequency)


def samples_lfm_signal(starting_frequency, frequency_at_pulse_time, sampling_frequency, pulse_duration):
    sampling_interval = get_sample_interval(sampling_frequency)
    samples_signal = get_samples(pulse_duration, sampling_frequency)
    bandwidth = get_bandwidth(starting_frequency, frequency_at_pulse_time)
    time_array = get_time_array(samples_signal, sampling_interval)

    result = create_empty_array(samples_signal)
    for k, _ in enumerate(time_array):
        k_dt = get_kdt(k, sampling_interval)
        result[k] = cos((((pi * bandwidth) / pulse_duration) * k_dt ** 2) + 2 * pi * starting_frequency * k_dt)

    return time_array, result


def samples_lfm_pulse(starting_frequency, frequency_at_pulse_time, sampling_frequency, pulse_duration):
    sampling_interval = get_sample_interval(sampling_frequency)
    samples_signal = get_samples(pulse_duration, sampling_frequency)
    bandwidth = get_bandwidth(starting_frequency, frequency_at_pulse_time)
    time_array = get_time_array(samples_signal, sampling_interval)

    result = create_empty_array(samples_signal)
    for k, _ in enumerate(time_array):
        k_dt = get_kdt(k, sampling_interval)
        result[k] = cos(((2 * pi * bandwidth * k_dt) - (pi * bandwidth / pulse_duration) * k_dt ** 2) + 2 * pi
                        * starting_frequency * k_dt)
    return time_array, result


def gen_sine(A, f, phi, fs, T):
    sampling_interval = get_sample_interval(fs)
    samples_signal = get_samples(T, fs)
    time_array = get_time_array(samples_signal, sampling_interval)

    result = create_empty_array(samples_signal)
    for n, _ in enumerate(result):
        result[n] = A * np.cos(2 * pi * f * n * sampling_interval + phi)
    return time_array, result


def get_same_result(f, n, sampling_interval, phi):
    return 2 * pi * f * n * sampling_interval + phi


def gen_complex_sine(f, N, A=1, phi=0, fs=1):
    sampling_interval = get_sample_interval(fs)

    result = create_empty_array(N, array_type='complex')
    for n, _ in enumerate(result):
        same_result = get_same_result(f, n, sampling_interval, phi)
        result[n] = A * complex(np.cos(same_result), np.sin(same_result))

    return result

def show_plot(time_array, result, time_array1=None, result2=None):
    fig, ax = plt.subplots()
    fig.tight_layout()
    plt.grid()
    plt.plot(time_array, result)
    if time_array1 is not None and result2 is not None:
        plt.plot(time_array1, result2)
    plt.show()


if __name__ == '__main__':
    # time, sinusoid = gen_sine(1.0, 10.0, 1.0, 100e2, 0.3)
    # show_plot(time, sinusoid)
    N = 120
    for k in range(0, 5):
        sinusoid1 = gen_complex_sine(-k / N, N)
        show_plot(np.arange(N), sinusoid1.real, np.arange(N), sinusoid1.imag)
    # (time, samples_one) = samples_lfm_signal(0, 100e3, 1000e3, 1.1e-3)
    # (time, samples_two) = samples_lfm_pulse(0, 100e3, 1000e3, 1.1e-3)
    # show_plot(time * 1000, samples_one)
    # show_plot(time * 1000, samples_two)
