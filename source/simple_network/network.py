import numpy as np
from typing import Union
from service.data_preparation import Service, data_item


class Network:
    def __init__(self, layers: dict, service: Service):
        self.num_of_layers = len(layers)
        self.service = service
        self.layers = layers
        self.weights = [self.init_matrix((i, j)) for i, j in zip(self.layers[:-1], self.layers[1:])]
        self.biases = [self.init_matrix((i, 1)) for i in self.layers[1:]]

    def forward_propagation(self, activation: np.array) -> np.array:
        for bias, weight in zip(self.biases, self.weights):
            activation = self.apply_sigmoid(np.dot(weight, activation))
        return activation

    def back_propagation(self, input_vector, expected_result) -> tuple:
        nabla_bias = [np.zeros(b.shape) for b in self.biases]
        nabla_weight = [np.zeros(w.shape) for w in self.weights]

        activation = input_vector
        activations = [activation]
        z_vectors = []

        # forward
        for bias, weight in zip(self.biases, self.weights):
            z_vector = np.dot(weight, activation) + bias
            z_vectors.append(z_vector)
            activation = self.apply_sigmoid(z_vector)
            activations.append(activation)

        # back
        delta = self.calculate_derivative_cost(activations[-1], expected_result) * \
                self.apply_derivative_sigmoid(z_vectors[-1])

        nabla_bias[-1] = delta
        nabla_weight[-1] = np.dot(delta, activations[-2].transpoce())

        # levels
        for layer_num in range(2, self.num_of_layers):
            z_vector = z_vectors[-layer_num]
            sd_vector = self.apply_derivative_sigmoid(z_vector)
            delta = np.dot(self.weights[-layer_num + 1].transpose(), delta) * sd_vector
            nabla_bias[-layer_num] = delta
            nabla_weight[-layer_num] = np.dot(delta, activations[-layer_num - 1].transpose())

        return nabla_bias, nabla_weight

    def update_network(self, training_set: dict, eta: float):
        nabla_bias = [np.zeros(bias.shape) for bias in self.biases]
        nabla_weight = [np.zeros(weight.shape) for weight in self.weights]

        for input_vector, expected_result in training_set:
            delta_nabla_bias, delta_nabla_weight = self.back_propagation(input_vector, expected_result)
            nabla_bias = [nb + dnb for nb, dnb in zip(nabla_bias, delta_nabla_bias)]
            nabla_weight = [nw + dnw for nw, dnw in zip(nabla_weight, delta_nabla_weight)]

        self.weights = [w - (eta / len(training_set)) * nw
                        for w, nw in zip(self.weights, nabla_weight)]
        self.biases = [b - (eta / len(training_set)) * nb
                       for b, nb in zip(self.biases, nabla_bias)]

    @staticmethod
    def init_matrix(dimensional: Union[tuple, int]):
        if len(dimensional) > 2:
            raise ValueError
        matrix = np.random.rand(*dimensional)
        return matrix

    @staticmethod
    def apply_sigmoid(vector: np.array):
        return 1 / (1 + np.exp(-vector))

    def apply_derivative_sigmoid(self, vector: np.array):
        return self.apply_sigmoid(vector) * (1 - self.apply_sigmoid(vector))

    @staticmethod
    def calculate_cost(expected_res, real_res):
        cost = 0
        for i in range(len(real_res)):
            cost += (expected_res[i] - real_res[i]) ** 2

        return cost

    @staticmethod
    def calculate_derivative_cost(expected_res: np.array, real_res: np.array) -> np.array:
        return real_res - expected_res


if __name__ == "__main__":
    net = Network([25, 5, 5])
