import numpy as np


def create_empty_array(size, array_type='double'):
    return np.zeros(size, dtype=array_type)


def find_index(array, value):
    for i, val in enumerate(array):
        if val >= value:
            return i
    return None


def get_real_from_tuple_array(array):
    return [x[0] for x in array]


def get_imaginary_from_tuple_array(array):
    return [[x[1] for x in array]]
