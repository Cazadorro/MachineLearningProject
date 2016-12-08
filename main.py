#!/bin/bash

"""
main file for Machine Learning Project
"""
import src.mapgeneration as map_gen
from src.mapextraction import read_map
from src.floyedwarshals import follow_floyd
from src.maputilities import to_1d_index, from_1d_index
from floyedwarshalls import optAPSP, opt2APSP
import numpy as np

def opt3APSP(cost_matrix, path_matrix, np_matrix = None):
    n = cost_matrix.shape[0]
    dist_matrix = np.copy(cost_matrix)
    next_matrix = np.copy(path_matrix)
    print(n)
    for k in range(n):
        print(k)
        for i in range(n):
            dist_ik = dist_matrix[i, k]
            if dist_ik != np.inf and i != k:
                for j in range(i):
                    new_dist = dist_ik + dist_matrix[k, j]
                    if dist_matrix[i, j] > new_dist:
                        dist_matrix[i, j] = new_dist
                        dist_matrix[i, j] = dist_matrix[j, i]
                        next_ik = next_matrix[i, k]
                        next_matrix[i, j] = next_matrix[i, k]
                        next_jk = next_matrix[j, k]
                        next_matrix[j, i] = next_matrix[j, k]
    return dist_matrix, next_matrix

#python setup.py build_ext --inplace
def main():
    for i in range(3):
        for j in range(3):
            x = to_1d_index((i,j), 3)
            y = from_1d_index(x, 3)
            print("i: ", i, " j: ", j)
            print("1d = ", x)
            print("2d = ", y)
    readm, readc = read_map('testdata/testmap2.txt')
    readc = np.asarray(readc)
    #generated = map_gen.generate_map(5, 5)
    #print(generated)
    test_map = np.array([[10, 10, 1], [10, 1, 10], [1, 10, 10]], dtype=np.uint8)
    print(test_map)
    cost_mat, path_matrix = map_gen.get_cost_matrix(readm)
    dist, next = opt3APSP(cost_mat, path_matrix)
    # for i in range(dist.shape[0]):
    #     for j in range(i):
    #         dist[j,i] = dist[i,j]
    #         next[j,i] = next[i,j]
    np.savetxt('path.out', cost_mat, delimiter=',', fmt='%4.0f')
    np.savetxt('cost.out', cost_mat, delimiter=',', fmt='%4.4f')
    np.savetxt('next.out', next, delimiter=',', fmt='%4.0f')
    np.savetxt('dist.out', dist, delimiter=',', fmt='%4.4f')

    print(readc.shape)
    follow_floyd(readc, next, (0, 0), (readc.shape[0] - 1, readc.shape[1] - 1))

    pass


if __name__ == "__main__":
    # run main program
    main()
