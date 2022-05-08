import numpy as np


def create_empty_array(size, array_type='double'):
    return np.zeros(size, dtype=array_type)


def find_index(array, value):
    for i, val in enumerate(array):
        if val >= value:
            return i
    return None


def generate_noise_array(samples):
    return np.random.randn(samples)


def add_zeroes(array, behind, front):
    return np.hstack([np.zeros(behind), array, np.zeros(front)])


def convert_to_complex(mas):
    output = create_empty_array(len(mas), array_type='complex')

    for k, number in enumerate(mas):
        output[k] = complex(number[0], number[1])

    return output
