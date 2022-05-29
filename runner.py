import numpy as np

from array_utils import *
from lfm import *
from plot import show_plot, show_multiplot

if __name__ == '__main__':
    # ЛЧМ сигнал во временной области (действительная составляющая)
    Fs = 125000  # 1e6
    T = 0.0011  # 2e-3
    dF = 126500  # 1e5
    wavelength = 5
    distance_to_target = 400.8e3
    speed_target = 98
    recurrence_interval = 0.0051
    ratio = -5
    time_receipt = get_time(distance_to_target)

    time_array, lfm_pulse = lfm_pulse(Fs, T, dF)
    # show_plot(time_array * 1000,
    #           [lfm_pulse.real],
    #           "ЛЧМ сигнал", "Время мс", "")

    reflected_signal, time_array_reflected = \
        generate_reflected_pulse(lfm_pulse, recurrence_interval, time_receipt, Fs)
    distance_reflected_signal = get_distance_array_from_time(time_array_reflected)
    # show_plot(distance_reflected_signal / 1000,
    #           [reflected_signal.real],
    #           "Отраженный сигнал без шума", "Дальность, км", f"SNR=0дБ")

    # # # Окно Хэмминга
    # # hamming = np.hamming(len(lfm_pulse))
    # # show_plot(np.arange(len(hamming)), [hamming], "Оконная функция Хэмминга", "Отсчеты")
    #
    # # ЛЧМ сигнал с весовой обработкой (действительная составляющая)
    weight_lfm = weight_signal_hamming(lfm_pulse)
    # show_plot(np.arange(len(weight_lfm)),
    #           [weight_lfm.real],
    #           "Опорный сигнал (действительная часть)", "Отсчеты")
    #
    # Интерполяция для уточнения максимального значения
    # compressed_signal = matched_filtering(weight_lfm, reflected_signal)
    # compressed_signal = log_scale(abs(compressed_signal))
    # pol_value, correct_distance = interpolation(compressed_signal, distance_reflected_signal, Fs)
    # print("polynomial = ", pol_value, "R = ", round(correct_distance / 1000, 2))
    # show_multiplot([distance_reflected_signal / 1000, correct_distance / 1000],
    #                [compressed_signal, pol_value],
    #                "Результат согласованной фильтрации",
    #                "Дальность, км", f"Дальность {round(correct_distance / 1000, 2)} км")

    # Доплеровская составляющая
    reflected_pulses = doppler_effect(reflected_signal, wavelength, speed_target, Fs, recurrence_interval, 512, ratio)
    reflected_pulses1 = doppler_effect(reflected_signal, wavelength, 5000, Fs, recurrence_interval, 512, ratio)
    distance_reflected_signals = get_distance_array_reflected_pulses(reflected_pulses,
                                                                     distance_reflected_signal,
                                                                     distance_to_target,
                                                                     speed_target, recurrence_interval)
    distance_reflected_signals1 = get_distance_array_reflected_pulses(reflected_pulses,
                                                                      distance_reflected_signal,
                                                                      distance_to_target,
                                                                      5000, recurrence_interval)
    # # Первый и последний импульс из пачки
    # show_multiplot([distance_reflected_signals[0] / 1000,
    #                 distance_reflected_signals[511] / 1000],
    #                [reflected_pulses[0].real,
    #                 reflected_pulses[511].real],
    #                "Первый и последний отраженный импульс", "Дальность км", f"SNR={ratio}дБ")

    # Сжатие первого и последнего сигнала по модулю
    compressed_signals = compress_signals(weight_lfm, reflected_pulses)
    compressed_signals1 = compress_signals(weight_lfm, reflected_pulses1)
    compressed_signals_abs = [abs(signal) for signal in compressed_signals]
    compressed_signals_abs1 = [abs(signal) for signal in compressed_signals1]
    # show_multiplot([distance_reflected_signals[0] / 1000,
    #                 distance_reflected_signals[511] / 1000],
    #                [log_scale(abs(compressed_signals[0])).real,
    #                 log_scale(abs(compressed_signals[511])).real],
    #                "Сжатие первого и последнего отраженного импульса",
    #                "Дистанция, км")

    # Амплитудное накопление
    final_signal = amplitude_accumulation(compressed_signals_abs, distance_reflected_signals, distance_reflected_signal)
    dist_last = find_close_index(final_signal, max(final_signal))
    final_signal1 = amplitude_accumulation(compressed_signals_abs1, distance_reflected_signals1,
                                           distance_reflected_signal)
    dist_last1 = find_close_index(final_signal1, max(final_signal1))
    show_plot(distance_reflected_signal / 1000,
              [log_scale(final_signal).real,
               log_scale(final_signal1).real],
              "Амплитудное накопление",
              "Дальность, км",
              f"Дистанция воздушного объекта {round(distance_reflected_signal[dist_last] / 1000, 2)} км\n"
              f"Дистанция воздушного объекта {round(distance_reflected_signal[dist_last1] / 1000, 2)} км",
              [speed_target, 5000])

    # # Измерение скорости и прочее
    # sliced_reflected_pulses, time = slice_reflected_pulses(compressed_signals, dist_last, recurrence_interval)
    # show_plot(time,
    #           [sliced_reflected_pulses.real,
    #            sliced_reflected_pulses.imag],
    #           f"Срез сигнала по отсчету {dist_last}",
    #           "Время, с")
    #
    # # Оконная обработка окном Блэкмана
    # weight_signal = weight_signal_blackman(sliced_reflected_pulses)
    # show_plot(np.arange(len(weight_signal)),
    #           [weight_signal.real,
    #            weight_signal.imag],
    #           "Оконная обработка",
    #           "Отсчеты")
    # #
    # cog, freq = cog_nak(weight_signal, recurrence_interval, wavelength)
    # radial_speed = get_radial_speed(cog, freq)
    # show_plot(freq,
    #           [log_scale(cog.real)],
    #           "Результат когерентного накопления",
    #           "Радиальная скорость м/с",
    #           f"Радиальная скорость = {round(radial_speed, 2)} м/с")

    # # # Без амплитудного накопления
    cog_naks, speed, index = without_amplitude(compressed_signals, recurrence_interval, wavelength)
    cog_naks1, speed1, index1 = without_amplitude(compressed_signals1, recurrence_interval, wavelength)

    show_multiplot([cog_naks[index][1], cog_naks1[index1][1]],
                   [log_scale(abs(cog_naks[index][0])).real,
                    log_scale(abs(cog_naks1[index1][0])).real],
                   "Когерентное накопление",
                   "Радиальная скорость, м/с", None, [speed_target, 5000])
    print(index, index1)
