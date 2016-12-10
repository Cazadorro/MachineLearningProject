#!/bin/bash

"""
main file for Machine Learning Project
"""
import src.mapgeneration as map_gen
from src.mapextraction import read_map
from floyedwarshalls import optimizedAPSP
from src.testgeneration import write_csv
import sys


# python cythonsrc/setup.py build_ext --inplace
def main():
    map_paths = sys.argv[1:]
    for path in map_paths:
        file_name = path.split('.')[0].split('/')[-1]
        print(file_name)
        readm, _ = read_map(path, '0', '1')
        cost_mat, path_mat = map_gen.get_cost_matrix(readm)
        dist, next_mat = optimizedAPSP(cost_mat, path_mat, readm)
        write_csv(file_name + "_next_paths.csv", next_mat)


if __name__ == "__main__":
    # run main program
    main()
