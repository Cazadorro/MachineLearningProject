#!/bin/bash
import numpy as np


def read_map(file_name):
    file = open(file_name, 'r')
    new_map = []
    char_map = []
    for line in file:
        new_line = []
        char_line = []
        for char in line:
            if char != '\n':
                print(char,  end='')
                temp_val = 0
                if char == '#':
                    temp_val = np.inf
                new_line.append(temp_val)
                char_line.append(char)
        print(len(line))

        new_map.append(new_line)
        char_map.append(char_line)
    new_map = np.array(new_map, dtype=float)
    file.close()
    return new_map, char_map
