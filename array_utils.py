import numpy as np


def create_empty_array(size, array_type='double'):
    return np.zeros(size, dtype=array_type)


def find_close_index(array, value):
    for i, val in enumerate(array):
        if val >= value:
            return i
    return None


def generate_noise_array(samples):
    return np.random.randn(samples).astype('complex', copy=False)


def add_zeroes(array, behind, front):
    return np.hstack([np.zeros(behind, dtype='complex'), array, np.zeros(front, dtype='complex')], )


def convert_to_complex(mas):
    output = create_empty_array(len(mas), array_type='complex')

    for k, number in enumerate(mas):
        output[k] = complex(number[0], number[1])

    return output


def get_real_from_tuple_array(array):
    return [x[0] for x in array]


def get_imaginary_from_tuple_array(array):
    return [[x[1] for x in array]]


def place(arr1, arr2, mark):
    output = np.copy(arr1)
    output[mark:len(arr2) + mark] = arr2
    return output

