from audio_reader import *
from dft import IDFT, DFT, gen_complex_sine, gen_mag_spec
from plot import show_plot
import numpy as np

if __name__ == '__main__':
    # # samplerate = get_samplerate('resources/kdt_437.wav')
    # show_wave('resources/kdt_437.wav')
    # # data = get_data('resources/kdt_437.wav')
    # show_amplitude_spec_wave('resources/kdt_437.wav')
    # show_log_amplitude_spec_wave('resources/kdt_437.wav')
    show_fragment_wave('resources/kdt_437.wav', 1.25, 1.37)

    # complex_sine = gen_complex_sine(2000.0, 500, fs=8000)
    # test_gen_mag_spec, f_array = gen_mag_spec(complex_sine, 8000)
    # show_plot(f_array, [test_gen_mag_spec])

    # test_array = np.array([1, 1, 1, 1])
    # dft = DFT(test_array)
    # print(dft)
    # idft = IDFT(dft)
    # print(idft)

    # print(dft)

    # test_array = np.array([1, -1, 1, -1])
    # print(np.around(DFT(test_array).real))
    # print(np.fft.fft(test_array))

    # (time, sinusoid) = gen_sine(1.0, 10.0, 1.0, 50.0, 0.1)
    # print(sinusoid)
    # print(time)
    # show_plot(time, [sinusoid])

    # N = 64
    # f1 = (np.arange(10. / 64, 11. / 64))
    # for f in f1:
    #     sinusoid1 = gen_complex_sine(f, N)
    #     dft = DFT(sinusoid1)
    #     fex = np.arange(N) / N * 1
    #     show_plot(fex, [dft.real, dft.imag, np.abs(dft)])

    # k = (np.arange(-64/2, 63/2))
    # x = gen_complex_sine(10. / 64, 64)
    # x1 = gen_complex_sine(20. / 64, 64)
    # dft = DFT(x.real)
    # dft1 = DFT(x1.real)
    # X = np.fft.fftshift(dft)
    # X1 = np.fft.fftshift(dft1)
    # show_plot(k, [np.abs(X), np.abs(X1) / 2])

    # (time, samples_one) = samples_lfm_signal(0, 100e3, 1000e3, 1.1e-3)
    # (time, samples_two) = samples_lfm_pulse(0, 100e3, 1000e3, 1.1e-3)
    # show_plot(time * 1000, samples_one)
    # show_plot(time * 1000, samples_two)
