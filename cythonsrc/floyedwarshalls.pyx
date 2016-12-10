#!python
#cython: language_level=3, boundscheck=False, wraparound=False
# distutils: language=c++

import numpy as np

def to_1d_index(index, width):
    return (index[0] * width) + index[1]


def from_1d_index(index, width):
    return index // width, index % width

def FloydAPSP(cost_matrix):
    # cost matrix must be square
    cdef int number_cells
    number_cells = cost_matrix.shape[0]
    dist_matrix = np.empty(cost_matrix.shape, dtype="float")
    path_matrix = np.full(cost_matrix.shape, -1, dtype="int")
    for index, value in np.ndenumerate(cost_matrix):
        # initializing the distance to the cost matrix values initially
        dist_matrix[index] = value

    print(number_cells)
    cdef int new_dist
    for k in range(number_cells):
        for i in range(number_cells):
            for j in range(number_cells):
                new_dist = dist_matrix[i, k] + dist_matrix[k, j]
                if new_dist < dist_matrix[i, j]:
                    dist_matrix[i, j] = new_dist
                    path_matrix[i, j] = k
    return dist_matrix, path_matrix

def optAPSP(cost_matrix, np_matrix = None):
    cdef int n
    n = cost_matrix.shape[0]
    dist_matrix = np.empty(cost_matrix.shape, dtype="float")
    next_matrix = np.full(cost_matrix.shape, -1, dtype="int")
    for index, value in np.ndenumerate(cost_matrix):
        # initializing the distance to the cost matrix values initially
        dist_matrix[index] = value
        next_matrix[index] = index[1]

    print(n)
    cdef int i, j, k
    cdef float dist_ik
    for k in range(n):
        print(k)
        for i in range(k):
            dist_ik = dist_matrix[i, k]
            if dist_ik != np.inf and i != k:
                for j in range(i):
                    dist_matrix[i, j] = min(dist_matrix[i, j], dist_ik + dist_matrix[j, k])
                    next_matrix[i, j] = next_matrix[i, k]
        for i in range(k, n):
            dist_ki = dist_matrix[k, i]
            if dist_ki != np.inf and i != k:
                for j in range(k):
                    dist_matrix[i, j] = min(dist_matrix[i, j], dist_ki + dist_matrix[j, k])
                    next_matrix[i, j] = next_matrix[k, i]
                for j in range(k, i):
                    dist_matrix[i, j] = min(dist_matrix[i, j], dist_ki + dist_matrix[k, j])
                    next_matrix[i, j] = next_matrix[k, i]
    return dist_matrix, next_matrix

def optimizedAPSP(cost_matrix, path_matrix, np_matrix):
    """
    optimized version of floyed warshals algorithm
    :param cost_matrix: cost matrix for every node in graph
    :param path_matrix: initial path connection matrix
    :param np_matrix: numpy matrix which cost and path matrix are based on
    :return: dist_matrix, optimized next_path matrix
    """

    cdef int n
    n = cost_matrix.shape[0]
    dist_matrix = np.copy(cost_matrix)
    next_matrix = np.copy(path_matrix)
    print(n)
    width = len(np_matrix[0])
    cdef int i, j, k
    for k in range(n):
        k2d = from_1d_index(k, width)
        if np_matrix[k2d] != np.inf:
            print(k)
            for i in range(n):
                i2d = from_1d_index(i, width)
                if np_matrix[i2d] != np.inf:
                    dist_ik = dist_matrix[i, k]
                    if dist_ik != np.inf and i != k:
                        for j in range(i):
                            old_dist = dist_matrix[i, j]
                            new_dist = dist_ik + dist_matrix[k, j]
                            if dist_matrix[i, j] > new_dist:
                                dist_matrix[i, j] = new_dist
                                dist_matrix[j, i] = dist_matrix[i, j]
                                next_matrix[i, j] = next_matrix[i, k]
                                next_matrix[j, i] = next_matrix[j, k]
    return dist_matrix, next_matrix