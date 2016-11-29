#!/bin/bash

"""
main file for Machine Learning Project
"""
import src.mapgeneration as map_gen
from src.mapextraction import read_map
from src.floyedwarshals import FloydAPSP, follow_floyd
import numpy as np


def main():
    readm, readc = read_map('testdata/testmap.txt')
    generated = map_gen.generate_map(5, 5)
    print(generated)
    test_map = np.array([[10, 10, 1], [10, 1, 10], [1, 10, 10]], dtype=np.uint8)
    print(test_map)
    cost_mat = map_gen.get_cost_matrix(readm)
    dist, path = FloydAPSP(cost_mat)
    follow_floyd(readc, path, (0, 0), (readc.shape[0] - 1, readc.shape[1] - 1))

    pass


if __name__ == "__main__":
    # run main program
    main()
