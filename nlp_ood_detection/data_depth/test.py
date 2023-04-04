#! /usr/bin/python3
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

    Returns:
        IRW: IRW object
    """
    if method == 'irw':
        return IRW(**kwargs)
    else:
        raise NotImplementedError(f'Method {method} not implemented yet!')


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Test the similarity measures')

    method_parser = parser.add_subparsers(
        dest=r'method', help='Chosen method', required=True)
    energy_parser = method_parser.add_parser(
        name=r'energy', help='Energy method')
    irw_parser = method_parser.add_parser(name=r'irw', help='IRW method')
    irw_parser.add_argument(
        '--n_samples', help='Number of samples', default=1000, type=int)
    irw_parser.add_argument(
        '--n_dim', help='Number of dimensions', default=2, type=int)

    args, _ = parser.parse_known_args()
    args = vars(args)

    x_train, x_grid, xx, yy = load_data()

    args['data_train'] = x_train

    scorer = get_method(**args)

    score = scorer.score(x_grid)

    plt.contourf(xx, yy, score.reshape(xx.shape), levels=20, cmap='viridis')
    plt.scatter(x_train[:, 0], x_train[:, 1],
                label='Train data', marker='x', c='r', s=50)
    plt.colorbar()
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.show()


if __name__ == '__main__':

    main()
