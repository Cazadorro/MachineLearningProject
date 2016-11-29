#!/bin/bash

import numpy as np
from src.maputilities import ii8, in_bounds, to_1d_index


def generate_map(width, height):
    """
    generates a matrix of uint8s stochastically with dimensions of height x width
    :param width:
    :param height:
    :return:
    """
    map_shape = (np.int(height), np.int(width))
    random_map = np.random.randint(ii8.min, ii8.max, size=map_shape, dtype=np.uint8)
    return random_map


def generate_adjacent_indices(point, shape, reach):
    adjacent_indices = []
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            if i != 0 or j != 0:
                new_point = (point[0] + i, point[1] + j)
                if in_bounds(new_point, shape):
                    adjacent_indices.append(new_point)
    return adjacent_indices


def get_cost_matrix(np_matrix, move_cost=1):
    number_cells = np_matrix.size
    cost_matrix = np.full((number_cells, number_cells), np.inf)
    matrix_width = np_matrix.shape[1]
    for index, cost in np.ndenumerate(np_matrix):
        adjacent_indices = generate_adjacent_indices(index, np_matrix.shape, 1)
        second_index = to_1d_index(index,  matrix_width)
        for adjacent_index in adjacent_indices:
            first_index = to_1d_index(adjacent_index,  matrix_width)
            cost_matrix[first_index, second_index] = (cost ** 2) * np.linalg.norm(
                np.array(index) - np.array(adjacent_index)) + move_cost
            temp = cost_matrix[first_index, second_index]
    for i in range(number_cells):
        # initializing diagonals to 0 (takes not cost to get to the same cell)
        cost_matrix[i, i] = 0
    return cost_matrix
