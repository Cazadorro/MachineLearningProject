#!/bin/bash
from math import cos, sin
import numpy as np
ray_cast_inc = 0.5

def getsubgrid(x1, y1, x2, y2, grid):
    return [item[x1:x2] for item in grid[y1:y2]]

def ray_cast(prob_map, start, theta, subgrid_dim):

    subx1 = start[1] - subgrid_dim[1] if subgrid_dim[1] <= start[1] else 0
    suby1 = start[0] - subgrid_dim[0] if subgrid_dim[0] <= start[0] else 0
    subx2 = start[1] - subgrid_dim[1] if subgrid_dim[1] <= start[1] else 0
    suby2 = start[0] - subgrid_dim[0] if subgrid_dim[0] <= start[0] else 0
    x = start[1] + ray_cast_inc * cos(theta)
    y = start[0] + ray_cast_inc * sin(theta)
