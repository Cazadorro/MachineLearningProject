#!/bin/bash
import numpy as np


def FloydAPSP(cost_matrix):
    # cost matrix must be square
    number_cells = cost_matrix.shape[0]
    dist_matrix = np.empty(cost_matrix.shape)
    path_matrix = np.full(cost_matrix.shape, -1)
    for index, x in np.ndenumerate(cost_matrix):
        # initializing the distance to the cost matrix values initially
        dist_matrix[index] = x
    for i in range(number_cells):
        # initializing diagonals to 0 (takes not cost to get to the same cell)
        dist_matrix[i, i] = 0

    for k in range(number_cells):
        for i in range(number_cells):
            for j in range(number_cells):
                new_dist = dist_matrix[i, k] + dist_matrix[k, j]
                if new_dist < dist_matrix[i, j]:
                    dist_matrix[i, j] = new_dist
                    path_matrix[i, j] = k
    return dist_matrix, path_matrix
