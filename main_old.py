#!/bin/bash

"""
main file for Machine Learning Project
"""
import src.mapgeneration as map_gen
from src.mapextraction import read_map
from src.floyedwarshals import follow_floyd
from src.maputilities import to_1d_index, from_1d_index
from floyedwarshalls import optimizedAPSP
from src.testgeneration import write_csv, read_csv
import numpy as np



#python cythonsrc/setup.py build_ext --inplace
def main():
    readm, readc = read_map('testdata/testmap.txt', '.', '#')
    readc = np.asarray(readc)
    # cost_mat, path_mat = map_gen.get_cost_matrix(readm)
    # dist, next = optimizedAPSP(cost_mat, path_mat, readm)
    # write_next_path_matrix("path_saved.csv", next)
    # np.savetxt('path.out', path_mat, delimiter=',', fmt='%4.0f')
    # np.savetxt('cost.out', cost_mat, delimiter=',', fmt='%4.1f')
    # np.savetxt('next.out', next, delimiter=',', fmt='%4.0f')
    # np.savetxt('dist.out', dist, delimiter=',', fmt='%4.1f')
    next = read_csv("path_saved.csv", int)

    tchar_map = follow_floyd(readc, next, (0, 0), (readc.shape[0] - 1, readc.shape[1] - 1))
    file = open('pathmap.txt', 'w')

    for charline in tchar_map:
        for char in charline:
            file.write(char)
        file.write("\n")
    file.close()


    pass


if __name__ == "__main__":
    # run main program
    main()
