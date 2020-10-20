import numpy as np
from typing import List, Tuple
from service.data_preparation import Service


class Network:
    def __init__(self, layers: List[int], num_of_categories: int, file_names: List[str]):
        """
            Инициализация объекта, представляющего нейросеть

        :param layers: список, длина которого представляет число слоев,
                            значение элемента - число нейронов в соответствующем слое
        :param num_of_categories: число типов распознаваемых образов
        :param file_names: список с именами файлов с данными для обучения
        """
        self.service = Service()
        self.file_names = file_names

        self.num_of_layers = len(layers)
        self.num_of_categories = num_of_categories
        self.layers = layers
        self.weights = [self.init_matrix((j, i)) for i, j in zip(self.layers[:-1], self.layers[1:])]
        self.biases = [self.init_matrix((i, 1)) for i in self.layers[1:]]

    def stochastic_gradient_descent(self, epochs: int, mini_training_set_size: int, eta: float) -> None:
        """
            Функция, реализующая стохастический градиентный спуск. Обучающая выборка разбивается на мини-пакеты,
        после чего в каждую эпоху обучения происходит усреднение вектора градиента по мини-пакетам с последующим
        обновлением весов

        :param epochs: число эпох обучения
        :param mini_training_set_size: размер пакета, на которые разбивается исходная выборка
        :param eta:  коэифициент сходимости
        :return:
        """

        training_set = net.service.get_training_set(self.file_names, self.num_of_categories)
        training_set_size = len(training_set)
        for epoch_num in range(epochs):
            mini_training_sets = [training_set[k:k + mini_training_set_size]
                                  for k in range(0, training_set_size, mini_training_set_size)]

            for mini_training_set in mini_training_sets:
                self.update_network(mini_training_set, eta)

    def update_network(self, training_set: List[float], eta: float) -> None:
        """
            Функция, осуществляющая вычисление усредненного вектора градиента по одному мини-пакету
        с последующим обновлением весов

        :param training_set: набор входных данных для обучения
        :param eta: коэфициент сходимости
        :return:
        """
        training_set_size = len(training_set)

        # вектора для градиента
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights]

        for input_vector, expected_result in training_set:
            input_vector_gradient_bias, input_vector_gradient_weight = \
                self.forward_and_back_propagation(input_vector, expected_result)

            # усредненение векторов с градиентами
            gradient_biases = [nb + ivnb for nb, ivnb in zip(gradient_biases, input_vector_gradient_bias)]
            gradient_weights = [nw + ivnw for nw, ivnw in zip(gradient_weights, input_vector_gradient_weight)]

        self.weights = [weight - (eta / training_set_size) * gradient_weight
                        for weight, gradient_weight in zip(self.weights, gradient_weights)]
        self.biases = [bias - (eta / training_set_size) * gradient_bias
                       for bias, gradient_bias in zip(self.biases, gradient_biases)]

    def recognition(self, activation: np.array) -> np.array:
        """
            Функция, осуществляющая прямое распространение для распознавание входного образа

        :param activation: входной вектор, представляющий данные для распознавания
        :return:
        """
        for bias, weight in zip(self.biases, self.weights):
            activation = self.apply_sigmoid(np.dot(weight, activation))
        return activation

    def forward_and_back_propagation(self, input_vector: np.array, expected_result: np.array) -> \
            Tuple[List[np.array], List[np.array]]:
        """
            Функция, осуществляющий алгоритм прямого распространения с последующим обратным
        распространением ошибки

        :param input_vector: вектор, представляющий входной образ
        :param expected_result: вектор, обозначающий категорию входного образа
        :return: вычисленный градиент функции стоимости по одному входному образу
        """
        input_vector_gradient_bias = [np.zeros(bias.shape) for bias in self.biases]
        input_vector_gradient_weight = [np.zeros(weight.shape) for weight in self.weights]

        activation = input_vector
        activations = [activation]
        z_vectors = []

        #  прямое распространение
        for bias, weight in zip(self.biases, self.weights):
            z_vector = np.dot(weight, activation) + bias
            z_vectors.append(z_vector)
            activation = self.apply_sigmoid(z_vector)
            activations.append(activation)

        self.service.append_cost_value(self.calculate_cost(expected_result, activations[-1]))

        # вычисление ошибки выходного слоя
        delta = self.calculate_derivative_cost(expected_result, activations[-1]) * \
            self.apply_derivative_sigmoid(z_vectors[-1])

        input_vector_gradient_bias[-1] = delta
        input_vector_gradient_weight[-1] = np.dot(delta, activations[-2].transpose())

        # обратное распространение ошибки
        for layer_num in range(2, self.num_of_layers):
            z_vector = z_vectors[-layer_num]
            sigmoid_derivative_vector = self.apply_derivative_sigmoid(z_vector)
            delta = np.dot(self.weights[-layer_num + 1].transpose(), delta) * sigmoid_derivative_vector
            input_vector_gradient_bias[-layer_num] = delta
            input_vector_gradient_weight[-layer_num] = np.dot(delta, activations[-layer_num - 1].transpose())

        return input_vector_gradient_bias, input_vector_gradient_weight

    @staticmethod
    def init_matrix(dimensional: tuple) -> np.array:
        """
            Функция, возвращающая случайную матрицу или вектор

        :param dimensional: число измерений и их величина
        :return:
        """
        if len(dimensional) > 2:
            raise ValueError
        matrix = np.random.rand(*dimensional)
        return matrix

    @staticmethod
    def apply_sigmoid(vector: np.array) -> np.array:
        """
            Сигмоидальная функция

        :param vector: входной вектор
        :return: результат применения функции
        """
        return 1 / (1 + np.exp(-vector))

    def apply_derivative_sigmoid(self, vector: np.array) -> np.array:
        """
        Производная сигмоидальной функция

        :param vector: входной вектор
        :return: результат применения функции
        """
        return self.apply_sigmoid(vector) * (1 - self.apply_sigmoid(vector))

    @staticmethod
    def calculate_cost(expected_res, real_res) -> float:
        """
            Вычисление функции стоимости

        :param expected_res: вектор, представляющий категорию
        :param real_res: реальный выход нейросети
        :return: величина ошибки
        """
        cost = 0
        for item in range(len(real_res)):
            cost += (float(expected_res[item]) - float(real_res[item])) ** 2
        return cost

    @staticmethod
    def calculate_derivative_cost(expected_res: np.array, real_res: np.array) -> np.array:
        """
        Вычисление производной функции стоимости

        :param expected_res: вектор, представляющий категорию
        :param real_res: реальный выход нейросети
        :return: величина ошибки
        """

        output = real_res - expected_res
        return output


if __name__ == "__main__":
    file_names_ = [f"category_{i}.txt" for i in range(1, 6)]
    net = Network([25, 10, 5], 5, file_names_)

    # # .png в файлы .txt
    # for i in range(1, 6):
    #     net.service.create_training_set(f"D:\DeepLearning\dataset\input_directory\category_{i}", i)

    net.stochastic_gradient_descent(30, 10, 3.5)

    net.service.show_plot(30, 10, 3.5)

    # тестирование нейросети
    for number in range(1, 6):
        print(f"Изображение категории: {number}")
        print(net.recognition(np.array(net.service.png_to_array(f"pixel_{number}.png"))))
