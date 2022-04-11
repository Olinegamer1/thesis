from scipy.io import wavfile
from dft import gen_mag_spec
from plot import show_plot
from array_utils import find_index

import numpy as np


def read_audio_file(filename):
    return wavfile.read(filename)


def get_samplerate(filename):
    return read_audio_file(filename)[0]


def get_data(filename):
    return read_audio_file(filename)[1]


def show_wave(filename):
    samplerate = get_samplerate(filename)
    data = get_data(filename)
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])
    show_plot(time, [data])


def show_fragment_wave(filename, start_time, end_time):
    samplerate = get_samplerate(filename)
    data = get_data(filename)
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])
    start_index = find_index(time, start_time)
    end_index = find_index(time, end_time)
    print(start_index, end_index)
    show_plot(time[start_index:end_index], [data[start_index:end_index]])


def show_amplitude_spec_wave(filename):
    samplerate = get_samplerate(filename)
    data = get_data(filename)
    wav_mag_spec, f_array = gen_mag_spec(data, samplerate)
    show_plot(f_array, [wav_mag_spec])


def show_log_amplitude_spec_wave(filename):
    samplerate = get_samplerate(filename)
    data = get_data(filename)
    wav_mag_spec, f_array = gen_mag_spec(data, samplerate)
    temp_array = 20 * np.log10(wav_mag_spec / max(wav_mag_spec))
    half = int(len(f_array) / 2)
    show_plot(f_array[:half], [temp_array[:half]])
