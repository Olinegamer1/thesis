from numpy import cos, sin
from signal_utils import *


def get_kdt(k, sampling_interval):
    return k * sampling_interval


def samples_lfm_signal(starting_frequency, frequency_at_pulse_time, sampling_frequency, pulse_duration):
    sampling_interval = get_sampling_interval(sampling_frequency)
    samples_signal = get_samples(pulse_duration, sampling_frequency)
    pulse_bandwidth = get_bandwidth(starting_frequency, frequency_at_pulse_time)
    time_array = get_time_array(samples_signal, sampling_interval)

    output = create_empty_array(samples_signal)
    for k, _ in enumerate(time_array):
        k_dt = get_kdt(k, sampling_interval)
        output[k] = cos((((pi * pulse_bandwidth) / pulse_duration) * k_dt ** 2) + 2 * pi * starting_frequency * k_dt)

    return time_array, output


def samples_lfm_pulse(starting_frequency, frequency_at_pulse_time, sampling_frequency, pulse_duration):
    sampling_interval = get_sampling_interval(sampling_frequency)
    samples_signal = get_samples(pulse_duration, sampling_frequency)
    bandwidth = get_bandwidth(starting_frequency, frequency_at_pulse_time)
    time_array = get_time_array(samples_signal, sampling_interval)

    output = create_empty_array(samples_signal)
    for k, _ in enumerate(time_array):
        k_dt = get_kdt(k, sampling_interval)
        output[k] = cos(((2 * pi * bandwidth * k_dt) - (pi * bandwidth / pulse_duration) * k_dt ** 2) + 2 * pi
                        * starting_frequency * k_dt)
    return time_array, output


def lfm_pulse(sampling_frequency, pulse_duration, pulse_bandwidth):
    sampling_interval = get_sampling_interval(sampling_frequency)
    samples_signal = get_samples(pulse_duration, sampling_frequency)
    time_array = get_time_array(samples_signal, sampling_interval)

    output = create_empty_array(samples_signal, array_type='object')
    for k, _ in enumerate(time_array):
        alpha = slew_rate(pulse_bandwidth, pulse_duration)
        k_dt = get_kdt(k, sampling_interval)
        temp_result = alpha * k_dt * (pulse_duration - k_dt)
        output[k] = (cos(temp_result), sin(temp_result))
    return time_array, output

