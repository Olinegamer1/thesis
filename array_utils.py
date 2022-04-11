import numpy as np


def create_empty_array(size, array_type='double'):
    return np.zeros(size, dtype=array_type)


def find_index(array, value):
    for i, val in enumerate(array):
        if val >= value:
            return i
    return None
