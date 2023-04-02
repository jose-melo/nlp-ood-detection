#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.datasets import load_iris
from data_depth_measures.IRW import IRW
import pandas as pd


def load_data():
    iris_data = load_iris()
    data_df = pd.DataFrame(
        iris_data['data'], columns=iris_data['feature_names'])
    X_train = data_df[["petal length (cm)",  "petal width (cm)"]].to_numpy()

    x = np.linspace(0, 8, 5)
    y = np.linspace(0, 3, 5)
    xx, yy = np.meshgrid(x, y)
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    return X_train, X_grid, xx, yy


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a script')
    parser.add_argument(
        '--n_samples', help='Number of samples', default=1000, type=int)

    args, _ = parser.parse_known_args()

    n_samples = args.n_samples
    n_dim = 2

    X_train, X_grid, xx, yy = load_data()

    irw = IRW(n_dim=n_dim, n_samples=n_samples)
    dirw = irw.score(X_train=X_train, X_test=X_grid)

    plt.contourf(xx, yy, dirw.reshape(xx.shape), levels=20, cmap='viridis')
    plt.scatter(X_train[:, 0], X_train[:, 1],
                label='Train data', marker='x', c='r', s=50)
    plt.colorbar()
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.show()
