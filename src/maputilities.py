#!/bin/bash
import numpy as np
from math import sqrt

ii8 = np.iinfo(np.uint8)


def in_bounds(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def to_1d_index(index, width):
    return (index[0] * width) + index[1]


def from_1d_index(index, width):
    return index // width, index % width


def get_dir_vector(start, end):
    negated = np.array(end - start)
    magnitude = sqrt(negated[0] ** 2 + negated[1] ** 2)
    return negated / magnitude
