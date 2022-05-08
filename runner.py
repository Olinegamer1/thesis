from array_utils import *
from plot import show_plot
from lfm import *

if __name__ == '__main__':
    # # ЛЧМ сигнал во временной области (действительная составляющая)
    Fs = 1e6  # 1e6
    T = 2e-3  # 2e-3
    dF = 1e5  # 1e5
    time_array, lfm_pulse = lfm_pulse(Fs, T, dF)
    lfm_pulse = lfm_pulse.real
    lfm_pulse_with_zeroes = add_zeroes(lfm_pulse, 500, 1000)
    show_plot(time_array * 1000, [lfm_pulse], "ЛЧМ сигнал", "Время мс", "")

    # # ЛЧМ сигнал с весовой обработкой (действительная составляющая)
    weight_lfm = weight_signal(lfm_pulse)
    show_plot(np.arange(len(weight_lfm)), [weight_lfm], "Опорный сигнал (действительная часть)", "Отсчеты")

    # # Смесь сигнала с шумом
    noise = generate_noise_array(1000)
    ratio = -5
    noisySignal, requested_noise = mix_SNR(lfm_pulse_with_zeroes, noise, ratio)

    show_plot(get_distance_array(noisySignal, Fs) * 1000, [noisySignal],
              "Отраженный сигнал (действительная часть)", "Дальность км", f"SNR={ratio}дБ")

    # # Согласованная фильтрация
    oporn_signal = weight_lfm
    otrajen_signal = noisySignal
    res = matched_filtering(oporn_signal, otrajen_signal, 1000)
    res = res[:len(requested_noise)]
    show_plot(get_distance_array(res, Fs) * 1000, [res], "Результат согласованной фильтрации", "Дальность км")
