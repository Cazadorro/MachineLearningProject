#!/bin/bash
import numpy as np


def read_map(file_name, empty, filled, ignored=('\n',)):
    """
    reads in a map made of '.' for empty and '#' filled
    :param file_name: file get map from
    :param empty: empty character
    :param filled: filled character
    :param ignored: ignored characters
    :return: new_map of 0 and np.inf, and character map read in
    """
    file = open(file_name, 'r')
    new_map = []
    char_map = []
    for line in file:
        new_line = []
        char_line = []
        for char in line:
            if char not in ignored:
                temp_val = 0
                if char == filled:
                    temp_val = np.inf
                elif char == empty:
                    temp_val = 0
                new_line.append(temp_val)
                char_line.append(char)
        new_map.append(new_line)
        char_map.append(char_line)
    new_map = np.array(new_map, dtype=float)
    file.close()
    return new_map, char_map