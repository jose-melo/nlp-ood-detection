#! /usr/bin/python3
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


class IRW(object):
    """Implementation of the Integrated ranked-weigthed (IRW) measure"""

    def __init__(self, n_dim: int = 2, n_samples: int = 1000):
        self.n_dim = n_dim
        self.n_samples = n_samples

    def __sample_sphere(self):
        """Sample from the unit sphere"""
        u = np.random.randn(self.n_dim, self.n_samples)
        u /= np.linalg.norm(u, axis=0)
        return u

    def score(self, X_train: ndarray, X_test: ndarray) -> ndarray:
        self.X_train = X_train
        self.X_test = X_test

        # Sample from the unitr sphere (Let's Monte Carlo it)
        sampled_points = self.__sample_sphere()

        ranking = np.zeros((self.X_test.shape[0], self.n_samples))

        # Projection of the train data on the sampled points  from the unit sphere
        X_train_proj = self.X_train @ sampled_points

        # Projection of the test data on the sampled points  from the unit sphere
        X_test_proj = self.X_test @ sampled_points

        # Calculate the ranking of the test data
        X_train_proj.sort(axis=0)
        for i in range(self.X_test.shape[0]):
            for j in range(self.n_samples):
                ranking[i, j] = np.argmin(
                    abs(X_train_proj[:, j] - X_test_proj[i, j]))

        # Normalize the ranking
        D_irw = ranking / len(self.X_train)

        # ID measure
        for i in range(len(D_irw)):
            for j in range(len(D_irw[i])):
                D_irw[i, j] = min(D_irw[i, j], 1 - D_irw[i, j])

        # Monte Carlo average
        D_irw = np.mean(D_irw, axis=1)

        return D_irw
