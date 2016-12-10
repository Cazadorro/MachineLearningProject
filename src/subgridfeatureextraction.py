#!/bin/bash
from math import cos, sin
import math
import numpy as np
from src.mapgeneration import ii8, in_bounds

ray_cast_inc = 0.5
prob_map_hit_threshold = int(ii8.max / 2)


def get_sub_grid(x1, y1, x2, y2, grid):
    return np.array([item[x1:x2+1] for item in grid[y1:y2+1]])


def cast_forward(x, y, theta):
    x += ray_cast_inc * cos(theta)
    y += ray_cast_inc * sin(theta)
    return x, y


def ray_cast(prob_map, start, theta, subgrid_dim):
    """
    returns the distance to the closest
    :param prob_map: probability map, type np.array(size=(n,m), dtype=np.uint8)
    :param start: start position to build subgrid around
    :param theta: angle of ray cast
    :param subgrid_dim: x,y of subgrid, centered on start
    :return:
    """
    x_sq_rad = (subgrid_dim[1]-1) // 2
    y_sq_rad = (subgrid_dim[0]-1) // 2
    subx1 = start[1] - x_sq_rad if x_sq_rad <= start[1] else 0
    suby1 = start[0] - y_sq_rad if y_sq_rad <= start[0] else 0
    subx2 = start[1] + x_sq_rad if (x_sq_rad + start[1]) < prob_map.shape[1]-1 else prob_map.shape[1] - 1
    suby2 = start[0] + y_sq_rad if (y_sq_rad + start[0]) < prob_map.shape[0]-1 else prob_map.shape[0] - 1

    # print(subx1)
    # print(suby1)
    # print(subx2)
    # print(suby2)
    center_point_x = x_sq_rad if subx1 != 0 else start[1]
    center_point_y = y_sq_rad if suby1 != 0 else start[0]
    new_start = np.array([center_point_y, center_point_x])
    sub_grid = get_sub_grid(subx1, suby1, subx2, suby2, prob_map)
    x = center_point_x + ray_cast_inc * cos(theta) + .0001
    y = center_point_y + ray_cast_inc * sin(theta) + .0001

    while in_bounds((int(round(y)), int(round(x))), sub_grid.shape):
        if sub_grid[int(round(y)), int(round(x))] > prob_map_hit_threshold:
            return np.linalg.norm(new_start - np.array(([y, x])))
        x, y = cast_forward(x, y, theta)
    # returning euclidian distance to found cast point.
    #np.inf
    return 1000


def get_cast_distances(prob_map, start, theta, subgrid_dim):
    angles = []
    current_angle = 0
    PI2 = 2 * math.pi
    while current_angle < PI2:
        angles.append(current_angle)
        current_angle += theta
    return [ray_cast(prob_map, start, angle, subgrid_dim) for angle in angles]


def sub_resolution_grid(np_grid, down_scaled_shape):
    scaled_shape = (np_grid[0] / down_scaled_shape[0], np_grid[1] / down_scaled_shape[1])
    scale_mat = np.array([[scaled_shape[0], 0], [0, scaled_shape[1]]])
    new_mat = np.empty(scaled_shape)
    for index in np_grid.shape:
        new_pos = scale_mat.dot(index)
        new_mat[new_pos] = np_grid[index]
    return new_mat
