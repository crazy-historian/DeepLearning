import numpy as np


def get_input_layers(filename):
    file = open(filename, 'r')
    str_input = file.read()
    input_layer = np.array(list(str_input))
    return input_layer


def apply_sigmoid(array):
    return 1 / (1 + np.exp(-array))


def init_matrix(*dimensional):
    if len(dimensional) > 2:
        raise ValueError
    matrix = np.random.rand(*dimensional)
    return matrix


def forward_propagate(matrix, input_layer, bias_vector):
    bias_vector = np.zeros(1)
    output_layer = apply_sigmoid(matrix * input_layer + bias_vector)
    return output_layer


if __name__ == "__main__":
    A = init_matrix(3)
