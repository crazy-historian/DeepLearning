import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from typing import Union
from random import shuffle
from collections import namedtuple

data_item = namedtuple("data_item", ["input", "category"])

# TODO: добавить функцию искажения входных образов


class Service:

    def __init__(self):
        self.cost_values = []

    def append_cost_value(self, value: float) -> None:
        """
            Добавить значение функции стоимости в список
        для дальнейшего построения графика

        :param value: значение функции стоимости
        :return:
        """
        self.cost_values.append(value)

    def create_training_set(self, directory_path: str, category: int) -> None:
        """
            Функция, осуществляющая преобразования множества изображений
        в один текстовый файл

        :param directory_path: путь до файла
        :param category: номер категории
        :return:
        """
        output_file = open(f"category_{category}.txt", 'w')

        for file_name in Path(directory_path).iterdir():
            array = self.png_to_array(file_name)
            for i in array:
                print(str(i), file=output_file, end=" ")
            else:
                print(file=output_file, end='\n')
        else:
            output_file.close()

    @staticmethod
    def png_to_array(filename: Union[str, Path]) -> list:
        """
            Преобразование изображения в текстовую строку

        :param filename: имя файла с изображением
        :return:
        """
        img = Image.open(filename).convert('RGBA')
        pixels = np.array(img)
        output_array = []
        alpha = 3
        for i in range(img.height):
            for j in range(img.height):
                if pixels[i][j][alpha] == 255:
                    output_array.append(1)
                else:
                    output_array.append(0)
        return output_array

    @staticmethod
    def get_training_set(file_names: list, quantity_of_categories: int) -> list:
        """
            Подготовка всего обучающего набора

        :param file_names: имена файлов с переведенными в текст изображениями
        :param quantity_of_categories: число категорий распознаваемых образов
        :return:
        """
        training_set = []
        category = -1
        for file_name in file_names:
            category += 1
            file = open(file_name, 'r')
            for line in file:
                activation_vector = np.array(list(map(float, line.rstrip().split(" ")))).reshape(-1, 1)
                category_vector = np.zeros(quantity_of_categories).reshape(-1, 1)
                category_vector[category] = 1
                training_set.append((activation_vector, category_vector))
            else:
                file.close()
        else:
            shuffle(training_set)

        return training_set[:100]

    def show_plot(self, num_of_epoch: int, mini_set_size: int, eta: float) -> None:
        """

        :param num_of_epoch: число эпох обучения
        :param mini_set_size: размер мини пакета
        :param eta: коэфициент сходимости
        :return:
        """
        figire, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.set_title("Функция стоимости С(w,b)")
        ax.set_ylabel("Значения функции стоимости")
        ax.set_xlabel("Итерации обучения")

        plt.plot(self.cost_values, c='blue', label=f"Число эпох: {num_of_epoch}, Размер пакета {mini_set_size}, "
                                                   f"Коэффициент сходимости: {eta}")
        plt.legend(loc="upper left")
        plt.show()
