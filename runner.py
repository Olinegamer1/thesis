from array_utils import *
from lfm import *
from plot import show_plot

if __name__ == '__main__':
    # ЛЧМ сигнал во временной области (действительная составляющая)
    Fs = 125000  # 1e6
    T = 0.0011  # 2e-3
    dF = 127000  # 1e5
    wavelength = 5
    distance_to_target = 400e3
    speed_target = 0
    recurrence_interval = 0.0051
    ratio = -15
    NN = 512

    time_receipt = get_time(distance_to_target)
    time_array, lfm_pulse = lfm_pulse(Fs, T, dF)
    show_plot(time_array * 1000,
              [lfm_pulse.real],
              "ЛЧМ сигнал", "Время мс", "")

    reflected_signal, time_array_reflected = \
        generate_reflected_pulse(lfm_pulse, T, recurrence_interval, time_receipt, Fs)
    distance_reflected_signal = get_distance_array(time_array_reflected)

    show_plot(distance_reflected_signal / 1000,
              [reflected_signal.real],
              "Отраженный сигнал без шума", "Дальность, км", f"SNR=0дБ")

    # ЛЧМ сигнал с весовой обработкой (действительная составляющая)
    weight_lfm = weight_signal(lfm_pulse)
    show_plot(np.arange(len(weight_lfm)),
              [weight_lfm.real],
              "Опорный сигнал (действительная часть)", "Отсчеты")

    # Смесь сигнала с шумом
    noise = generate_noise_array(1000)
    noisy_signal, requested_noise = mix_SNR(reflected_signal, noise, ratio)
    show_plot(distance_reflected_signal / 1000, [noisy_signal.real],
              "Смесь сигнала с шумом", "Дальность, км", f"SNR={ratio}дБ")

    # Доплеровская составляющая
    reflected_pulses, distances = doppler_effect(noisy_signal, wavelength, speed_target,
                                                 Fs, recurrence_interval, NN, distance_to_target)

    # Первый и последний импульс из пачки
    show_plot(distance_reflected_signal,
              [reflected_pulses[0].real, reflected_pulses[1].real,
               reflected_pulses[2].real, reflected_pulses[3].real],
              "Первый и последний отраженный импульс", "Дальность км", f"SNR={ratio}дБ")

    # Сжатие первого и последнего сигнала по модулю
    compressed_signals = compress_signals(weight_lfm, reflected_pulses)
    dist = find_close_index(compressed_signals[511].real, max(compressed_signals[511]))
    show_plot(distance_reflected_signal / 1000,
              [compressed_signals[0].real, compressed_signals[250].real,
               compressed_signals[400].real, compressed_signals[511].real],
              "Сжатие первого и последнего отраженного импульса",
              "Дальность, км")

    # Амплитудное накопление
    final_signal = amplitude_accumulation(compressed_signals)
    dist_last = find_close_index(final_signal.real, max(final_signal.real))
    show_plot(distance_reflected_signal / 1000,
              [final_signal.real],
              "Амплитудное накопление",
              "Дальность, км",
              f"R = {round(distance_reflected_signal[dist_last] / 1000, 2)}")
