import numpy as np

from numpy import log10, max


def create_empty_array(size, array_type='double'):
    return np.zeros(size, dtype=array_type)


def find_close_index(array, value):
    for i, val in enumerate(array):
        if val == value:
            return i
        elif val > value:
            return i - 1
    return None


def find_close_index_ampl(array, value):
    min_index = find_close_index(array, value)
    left_border = array[min_index]
    right_border = array[min_index + 1]

    first_abs = abs(value - left_border)
    second_abs = abs(right_border - value)

    if first_abs < second_abs:
        return min_index
    else:
        return min_index + 1


def generate_noise_array(samples):
    return np.random.randn(samples).astype('complex', copy=False)


def place(arr1, arr2, mark):
    output = np.copy(arr1)
    output[mark:len(arr2) + mark] = arr2
    return output


def log_scale(array):
    array = array / max(array.real)
    return 20 * log10(array.real)


def slice_range(arrays, index):
    output = create_empty_array(len(arrays), array_type='complex')
    for k, array in enumerate(arrays):
        output[k] = array[index]
    return output
