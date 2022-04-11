import numpy as np

from signal_utils import get_sample_interval, get_samples, get_time_array
from array_utils import create_empty_array
from numpy import pi, absolute


def gen_sine(A, f, phi, fs, T):
    sampling_interval = get_sample_interval(fs)
    samples_signal = get_samples(T, fs)
    time_array = get_time_array(samples_signal, sampling_interval)

    sine = create_empty_array(samples_signal)
    for n, _ in enumerate(sine):
        sine[n] = A * np.cos(2 * pi * f * n * sampling_interval + phi)
    return time_array, sine


def gen_complex_sine(f, N, A=1., phi=0., fs=1.):
    sampling_interval = get_sample_interval(fs)

    complex_sine = create_empty_array(N, array_type='complex')
    for n, _ in enumerate(complex_sine):
        temp_calculate = 2 * np.pi * f * n * sampling_interval + phi
        complex_sine[n] = A * complex(np.cos(temp_calculate), np.sin(temp_calculate))

    return complex_sine


def DFT(x):
    output = create_empty_array(len(x), array_type='complex')

    for n, _ in enumerate(x):
        complex_sine = gen_complex_sine(-n / len(x), len(x))
        output[n] = np.dot(x, complex_sine)
    return output


def IDFT(X):
    output = create_empty_array(len(X), array_type='complex')

    for n, _ in enumerate(X):
        complex_sine = gen_complex_sine(n / len(X), len(X))
        output[n] = np.dot(X, complex_sine) / len(X)
    return output


def gen_mag_spec(x, fs):
    f_array = np.arange(len(x)) * fs / len(x)
    return absolute(np.fft.fft(x)), f_array
