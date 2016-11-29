#!/bin/bash

"""
main file for Machine Learning Project
"""
import src.mapgeneration as map_gen
from src.floyedwarshals import FloydAPSP
import numpy as np


def main():
    generated = map_gen.generate_map(5, 5)
    print(generated)
    test_map = np.array([[10, 10, 1], [10, 1, 10], [1, 10, 10]], dtype=np.uint8)
    print(test_map)
    cost_mat = map_gen.get_cost_matrix(generated)
   # print(cost_mat)
    for i in range(cost_mat.shape[0]):
        print(cost_mat[i])
    dist, path = FloydAPSP(cost_mat)
    print(dist)
    print(path)
    pass


if __name__ == "__main__":
    # run main program
    main()
