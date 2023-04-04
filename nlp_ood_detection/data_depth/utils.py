import argparse
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import ndarray
from sklearn.datasets import load_iris
from nlp_ood_detection.data_depth.similarity_scorer import IRW, EnergyBased, Mahalanobis, MSP


def load_data() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Load Iris dataset

    Returns:
        _type_: _description_
    """
    iris_data = load_iris()
    data_df = pd.DataFrame(
        iris_data['data'], columns=iris_data['feature_names'])
    x_train = data_df[["petal length (cm)",  "petal width (cm)"]].to_numpy()

    x = np.linspace(0, 8, 5)
    y = np.linspace(0, 3, 5)
    xx, yy = np.meshgrid(x, y)
    x_grid = np.c_[xx.ravel(), yy.ravel()]
    return x_train, x_grid, xx, yy


def get_method(method: str, **kwargs) -> IRW:
    """Get method

    Args:
        method (str): Chosen method

    """
    if method == 'irw':
        return IRW(**kwargs)
    else:
        raise NotImplementedError(f'Method {method} not implemented yet!')
