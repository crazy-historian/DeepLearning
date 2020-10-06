from random import seed, randrange, random

import numpy as np


def get_input_layer(filename):
    file = open(filename, 'r')
    str_input = file.read()
    input_layer = np.array(list(str_input))
    return input_layer


# TODO: use *kwargs to modify this function for vector generation
def init_matrix(num_of_rows, num_of_column):
    matrix = np.random.rand(num_of_rows, num_of_column)
    return matrix


if __name__ == "__main__":
    W = init_matrix(25, 5)
    print(W)
