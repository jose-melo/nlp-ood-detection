import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.datasets import load_iris

from nlp_ood_detection.data_depth.similarity_scorer import (
    IRW,
    EnergyBased,
)


def load_data(
    x_min: float = 0,
    x_max: float = 8,
    y_min: float = 0,
    y_max: float = 3,
    grid_size: int = 5,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Load Iris dataset

    Returns:
        _type_: _description_
    """
    iris_data = load_iris()
    df = pd.DataFrame(iris_data["data"], columns=iris_data["feature_names"])
    df["target"] = iris_data["target"]
    x_train = df[["petal length (cm)", "petal width (cm)"]][df["target"] <= 1]
    x_train = x_train.to_numpy()
    y_train = df["target"][df["target"] <= 1].to_numpy()

    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x, y)
    x_grid = np.c_[xx.ravel(), yy.ravel()]
    return x_train, y_train, x_grid, xx, yy


def get_method(method: str, **kwargs) -> IRW:
    """Get method

    Args:
        method (str): Chosen method

    """
    if method == "irw":
        return IRW(**kwargs)
    if method == "energy":
        return EnergyBased(**kwargs)
    else:
        not_implemented = f"Method {method} not implemented yet!"
        raise NotImplementedError(not_implemented)
