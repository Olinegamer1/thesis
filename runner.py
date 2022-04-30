import numpy as np

from audio_reader import *
from dft import IDFT, DFT, gen_complex_sine, gen_mag_spec
from array_utils import *
from plot import show_plot
from lfm import *

if __name__ == '__main__':
    # ЛЧМ сигнал во временной области (действительная составляющая)
    Fs = 1e6
    T = 2e-3
    dF = 1e5
    time_array, pulse = lfm_pulse(Fs, T, dF)
    pulse = get_real_from_tuple_array(pulse)
    pulse = add_zeroes(pulse, 2000)
    # show_plot(np.arange(len(pulse)), [pulse])



    # Амплитудный спектр ЛЧМ сигнала
    # mag_pulse, f_array = gen_mag_spec(pulse, Fs)
    # temp_array = 20 * np.log10(mag_pulse / max(mag_pulse))
    # show_plot(f_array, [mag_pulse])

    # Окно Хэмминга во временной области
    # window_hamming = hamming_window(pulse)
    # show_plot(np.arange(len(window_hamming)), [window_hamming])

    # ЛЧМ сигнал с весовой обработкой (действительная составляющая)
    # weight_lfm = weight_signal(pulse, window_hamming)
    # show_plot(time_array, [get_real_from_tuple_array(weight_lfm)])

    # Амплитудный спектр ЛЧМ с весовой обработкой
    # mag_weight_pulse, f_array = gen_mag_spec(get_real_from_tuple_array(weight_lfm), Fs)
    # temp_array = 20 * np.log10(mag_weight_pulse / max(mag_weight_pulse))
    # show_plot(np.arange(len(f_array)), [temp_array.real])

    # Смесь сигнала с шумом
    noise = generate_noise_array(len(pulse))

    lfm_norm = norm(pulse)
    noise_norm = norm(noise)
    signal_mult = signal_multiplier(20)

    noise_scale = noise_signal_normalization(lfm_norm, noise_norm, noise, signal_mult)
    signal_scale = signal_scale(pulse, noise_norm, lfm_norm, signal_mult)
    add_sig_mix = additive_signal_mixture(signal_scale, noise_scale)
    show_plot(np.arange(len(add_sig_mix)), [add_sig_mix])

