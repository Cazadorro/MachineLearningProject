#!/bin/bash

"""
main file for Machine Learning Project
"""
import src.mapgeneration as map_gen
from src.mapextraction import read_map
from src.floyedwarshals import follow_floyd
from floyedwarshalls import optimizedAPSP
from src.testgeneration import write_csv, read_csv, gen_n_test
import numpy as np
import sys


# python cythonsrc/setup.py build_ext --inplace
def main():
    args = sys.argv[1:]
    map_path = args[0]
    next_path = args[1]
    n_points = int(args[2])
    readm, _ = read_map(map_path, '0', '1')
    next_mat = read_csv(next_path, int)
    test_x, test_y = gen_n_test(readm, next_mat, n_points, (9,9), 16)
    file_name = map_path.split('.')[0].split('/')[-1]
    write_csv(file_name+"_x_data.csv", test_x)
    write_csv(file_name+"_y_data.csv", test_y)



if __name__ == "__main__":
    # run main program
    main()
