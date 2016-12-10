#!/bin/bash
import numpy as np
from copy import deepcopy
from src.maputilities import to_1d_index, from_1d_index

def follow_floyd(char_map, path_matrix, start, end):
    """
    follow the path matrix and mark it on a character map
    :param char_map: character representation of map
    :param path_matrix: path matrix, format, location to move to to get from [i->j]
    :param start: start row column position
    :param end: end row column position
    :return:
    """
    tchar_map = deepcopy(char_map)
    width = len(char_map[0])
    current = start
    path_array = []
    path_array.append(current)
    end1d = to_1d_index(end, width)
    tchar_map[current[0]][current[1]] = '*'
    #print(tchar_map)
    while current != end:

        current1d = to_1d_index(current, width)
        next1d = path_matrix[current1d, end1d]
        current_t = from_1d_index(next1d, width)
        assert next1d != -1, "ERROR, current shouldn't be -1 current = {} {}".format(current1d, current)
        current = current_t
        path_array.append(current)
        tchar_map[current[0]][current[1]] = '*'
        #print(tchar_map)
        print(current)
    return tchar_map