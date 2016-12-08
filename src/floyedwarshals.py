#!/bin/bash
import numpy as np
from copy import deepcopy
from src.maputilities import to_1d_index, from_1d_index


def FloydAPSP(cost_matrix):
    # cost matrix must be square
    number_cells = cost_matrix.shape[0]
    dist_matrix = np.empty(cost_matrix.shape)
    path_matrix = np.full(cost_matrix.shape, -1)
    for index, value in np.ndenumerate(cost_matrix):
        # initializing the distance to the cost matrix values initially
        dist_matrix[index] = value
    print(number_cells)
    for k in range(number_cells):
        print(k)
        for i in range(number_cells):
            for j in range(number_cells):
                new_dist = dist_matrix[i, k] + dist_matrix[k, j]
                if new_dist < dist_matrix[i, j]:
                    dist_matrix[i, j] = new_dist
                    path_matrix[i, j] = k
    return dist_matrix, path_matrix


def follow_floyd(char_map, path_matrix, start, end):
    tchar_map = deepcopy(char_map)
    width = len(char_map[0])
    current = start
    path_array = []
    path_array.append(current)
    end1d = to_1d_index(end, width)
    tchar_map[current[0]][current[1]] = '*'
    print(tchar_map)
    while current != end:

        current1d = to_1d_index(current, width)
        next1d = path_matrix[current1d, end1d]
        current_t = from_1d_index(next1d, width)
        assert next1d != -1, "ERROR, current shouldn't be -1 current = {} {}".format(current1d, current)
        current = current_t
        path_array.append(current)
        tchar_map[current[0]][current[1]] = '*'
        print(tchar_map)