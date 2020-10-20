import numpy as np
from typing import Union
from service.data_preparation import Service


class Network:
    def __init__(self, layers: list, num_of_categoris: int, file_names: list):
        self.num_of_layers = len(layers)
        self.num_of_categories = num_of_categoris
        self.service = Service()
        self.layers = layers
        self.file_names = file_names
        self.weights = [self.init_matrix((j, i)) for i, j in zip(self.layers[:-1], self.layers[1:])]
        self.biases = [self.init_matrix((i, 1)) for i in self.layers[1:]]

    def stochastic_gradient_descent(self, epochs, mini_training_set_size, eta):
        training_set = net.service.get_training_set(self.file_names, self.num_of_categories)
        training_set_size = len(training_set)
        for epoch_num in range(epochs):
            mini_training_sets = [training_set[k:k + mini_training_set_size]
                                  for k in range(0, training_set_size, mini_training_set_size)]

            for mini_training_set in mini_training_sets:
                self.update_network(mini_training_set, eta)

    def update_network(self, training_set: list, eta: float):
        training_set_size = len(training_set)

        # vectors for gradient
        nabla_biases = [np.zeros(bias.shape) for bias in self.biases]
        nabla_weights = [np.zeros(weight.shape) for weight in self.weights]

        for input_vector, expected_result in training_set:
            input_vector_nabla_bias, input_vector_nabla_weight = \
                self.forward_and_back_propagation(input_vector, expected_result)

            # vector of average nabla vectors
            nabla_biases = [nb + ivnb for nb, ivnb in zip(nabla_biases, input_vector_nabla_bias)]
            nabla_weights = [nw + ivnw for nw, ivnw in zip(nabla_weights, input_vector_nabla_weight)]

        self.weights = [weight - (eta / training_set_size) * nabla_weight
                        for weight, nabla_weight in zip(self.weights, nabla_weights)]
        self.biases = [bias - (eta / training_set_size) * nabla_bias
                       for bias, nabla_bias in zip(self.biases, nabla_biases)]

    def forward_propagation(self, activation: np.array) -> np.array:
        for bias, weight in zip(self.biases, self.weights):
            activation = self.apply_sigmoid(np.dot(weight, activation))
        return activation

    def forward_and_back_propagation(self, input_vector, expected_result) -> tuple:
        input_vector_nabla_bias = [np.zeros(bias.shape) for bias in self.biases]
        input_vector_nabla_weight = [np.zeros(weight.shape) for weight in self.weights]

        activation = input_vector
        activations = [activation]
        z_vectors = []

        # forward
        for bias, weight in zip(self.biases, self.weights):
            z_vector = np.dot(weight, activation) + bias
            z_vectors.append(z_vector)
            activation = self.apply_sigmoid(z_vector)
            activations.append(activation)

        # error calculation for output activation
        delta = self.calculate_derivative_cost(expected_result, activations[-1]) * self.apply_derivative_sigmoid(z_vectors[-1])

        # TODO: обернуть в функцию ?
        input_vector_nabla_bias[-1] = delta
        input_vector_nabla_weight[-1] = np.dot(delta, activations[-2].transpose())

        # back propagation
        for layer_num in range(2, self.num_of_layers):
            z_vector = z_vectors[-layer_num]
            sigmoid_derivative_vector = self.apply_derivative_sigmoid(z_vector)
            delta = np.dot(self.weights[-layer_num + 1].transpose(), delta) * sigmoid_derivative_vector
            input_vector_nabla_bias[-layer_num] = delta
            input_vector_nabla_weight[-layer_num] = np.dot(delta, activations[-layer_num - 1].transpose())

        return input_vector_nabla_bias, input_vector_nabla_weight

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

    # @staticmethod
    # def calculate_cost(expected_res, real_res):
    #     cost = 0
    #     for i in range(len(real_res)):
    #         cost += (expected_res[i] - real_res[i]) ** 2
    #
    #     return cost

    @staticmethod
    def calculate_derivative_cost(expected_res: np.array, real_res: np.array) -> np.array:
        output = real_res - expected_res
        return output


if __name__ == "__main__":
    file_names_ = ["category_1.txt", "category_2.txt", "category_3.txt", "category_4.txt", "category_5.txt"]
    net = Network([25, 10, 5], 5, file_names_)

    # net.service.create_training_set("D:\DeepLearning\dataset\input_directory\category_1", 1)
    # net.service.create_training_set("D:\DeepLearning\dataset\input_directory\category_2", 2)
    # net.service.create_training_set("D:\DeepLearning\dataset\input_directory\category_3", 3)
    # net.service.create_training_set("D:\DeepLearning\dataset\input_directory\category_4", 4)
    # net.service.create_training_set("D:\DeepLearning\dataset\input_directory\category_5", 5)

    net.stochastic_gradient_descent(30, 5, 3.0)
    for i in range(1, 6):
        print(f"Изображение категории: {i}")
        print(net.forward_propagation(np.array(net.service.png_to_array(f"pixel_{i}.png"))))
