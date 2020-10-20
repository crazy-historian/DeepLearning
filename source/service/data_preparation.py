from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union
from random import shuffle
from collections import namedtuple

data_item = namedtuple("data_item", ["input", "category"])


class Service:

    @staticmethod
    def get_training_set(file_names: list, quantity_of_categories: int) -> list:
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

    def create_training_set(self, directory_path: str, category: int):
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
    def png_to_array(filename: Union[str, Path]):
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
