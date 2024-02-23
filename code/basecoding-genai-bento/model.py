from typing import List

import numpy as np


def my_python_model(input_list: List[int]) -> List[int]:
    return np.square(np.array(input_list))