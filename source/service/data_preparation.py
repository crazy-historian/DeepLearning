from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union
from collections import namedtuple

data_item = namedtuple("data_item", ["input", "category"])


class Service:

    def get_training_set(self, directory_path: str, category: int, quantity_of_categories: int) -> list:
        category_vector = np.zeros(quantity_of_categories)
        training_set = []
        for file_name in Path(directory_path).iterdir():
            activation_vector = self.png_to_array(file_name)
            category_vector[category] = 1
            training_set.append(data_item(activation_vector, category_vector))

        return training_set

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
        return np.array([output_array])
