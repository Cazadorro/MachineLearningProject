#!/bin/bash

import numpy as np
import src.subgridfeatureextraction as sgfe
from floyedwarshalls import optimizedAPSP
from src.mapgeneration import get_cost_matrix
import random
from src.maputilities import get_dir_vector, to_1d_index, from_1d_index
import math
import csv


def get_empty(np_matrix, empty_threshold=0):
    indexes = []
    for index, value in np.ndenumerate(np_matrix):
        if value <= empty_threshold:
            indexes.append(index)
    return indexes


def gen_n_points(empty_indices, num_indices):
    start_pts = []
    end_pts = []
    for i in range(num_indices):
        start_pts.append(random.choice(empty_indices))
        end_pts.append(random.choice(empty_indices))
    return start_pts, end_pts


def write_csv(file_name, np_matrix):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(np_matrix)


def read_csv(file_name, type_conv):
    np_matrix = []
    with open(file_name, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            np_matrix.append([type_conv(r) for r in row])
    return np.array(np_matrix)


def gen_n_test(np_matrix, next_path, num_points, sub_grid_size, num_angle):
    theta = (2 * math.pi) / num_angle
    width = np_matrix.shape[1]
    empty_indices = get_empty(np_matrix)
    start_pts, end_pts = gen_n_points(empty_indices, num_points)
    input_x = []
    expected_y = []
    sgs_rad = (sub_grid_size[0]) / 2
    for i in range(num_points):
        start = np.array(start_pts[i])
        end = np.array(end_pts[i])
        si = to_1d_index(start, width)
        gj = to_1d_index(end, width)
        nk = next_path[si, gj]
        next_point = from_1d_index(nk, width)
        y_line = get_dir_vector(start, next_point).reshape(1, 2)
        x_line = np.array(sgfe.get_cast_distances(np_matrix, start, theta, sub_grid_size))
        x_line = [(x/sgs_rad if x > 0 else -1) for x in x_line]
        input_x.append(x_line + [start[0]/width, start[1]/width] + [end[0]/width, end[1]/width])
        expected_y.append([y_line[0,0]]+[y_line[0,1]])
    return input_x, expected_y
