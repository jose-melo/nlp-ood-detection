from abc import ABC, abstractmethod
from typing import TypedDict, Union

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from torch.nn import Module
from torch.nn.functional import softmax


class SimilarityScorerBase(ABC):
    """Base class for similarity scorers"""

    def __init__(
        self,
        x_train: ndarray,
        y_train: ndarray | None = None,
        model: Module | None = None,
        features: ndarray | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.x_train = x_train
        self.model = model
        self.y_train = y_train
        self.x_test = None
        self.features = features

    @abstractmethod
    def score(self, x: ndarray, **kwargs) -> ndarray:
        """Calculate the similarity score
        :params: X (ndarray): input value
        :return: ndarray: the similarity score
        """


class Mahalanobis(SimilarityScorerBase):
    """Mahalanobis distance based similarity scorer"""

    def __init__(
        self,
        x_train: ndarray,
        y_train: ndarray | None = None,
        model: Module | None = None,
        feature: ndarray | None = [0, 1],
        **kwargs,
    ) -> None:
        super().__init__(x_train, y_train, model)
        self.feature = feature

    def score(self, x: ndarray, **kwargs) -> ndarray:
        x_test = x
        x_train = self.x_train

        x_test = x_test - np.mean(x_train, axis=0)

        cov = np.cov(x_train, rowvar=False)

        inv_cov = np.linalg.inv(cov)

        mahalanobis = np.dot(
            x_test,
            inv_cov,
        )
        mahalanobis = np.dot(mahalanobis, x_test.T)
        mahalanobis = np.diag(mahalanobis)

        return mahalanobis


class MSP(SimilarityScorerBase):
    """Implementation of the Maximum similarity probability (MSP) measure"""

    def __init__(
        self,
        x_train: ndarray,
        y_train: ndarray | None = None,
        model: Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(x_train, y_train, model)

    def score(self, x: ndarray, logits: ndarray, **kwargs) -> ndarray:
        self.x_test = x
        if logits is None:
            with torch.no_grad():
                logits = self.model(self.x_train, self.y_train, self.x_test).cpu()

        prediction = softmax(logits, dim=1)
        prediction = prediction.max(axis=1).values
        msp = 1 - prediction.numpy()

        return msp


class EnergyBased(SimilarityScorerBase):
    """Implementation of the Energy-based measure"""

    def __init__(
        self,
        x_train: ndarray,
        y_train: ndarray | None = None,
        model: Module | None = None,
        temperature: float = 1,
        use_logits: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(x_train, y_train, model)

        self.T = temperature
        self.use_logits = use_logits

    def __lse(self, x: ndarray) -> ndarray:
        """Implementation of the log-sum-exp function"""
        return -self.T * np.log(np.sum(np.exp(x / self.T), axis=1))

    def score(self, x: ndarray, logits: ndarray, **kwargs) -> ndarray:
        if logits is None:
            with torch.no_grad():
                logits = self.model(self.x_train, self.y_train, x).cpu()

        prediction = softmax(logits, dim=1).numpy()
        energy = self.__lse(prediction)

        return energy


class IRW(SimilarityScorerBase):
    """Implementation of the Integrated ranked-weigthed (IRW) measure"""

    def __init__(
        self,
        x_train: ndarray,
        y_train: ndarray | None = None,
        model: Module | None = None,
        num_dim: int = 2,
        num_samples: int = 1000,
        feature: list | None = [0, 1],
        **kwargs,
    ):
        super().__init__(x_train, y_train, model, feature)
        self.num_dim = num_dim
        self.num_samples = num_samples
        self.feature = feature

    def __sample_sphere(self):
        """Sample from the unit sphere"""
        generator = np.random.Generator(np.random.PCG64())
        u = generator.standard_normal((self.num_dim, self.num_samples))
        u /= np.linalg.norm(u, axis=0)
        return u

    def score(self, x: ndarray, **kwargs) -> ndarray:
        x_test = x
        x_train = self.x_train

        sampled_points = self.__sample_sphere()

        ranking = np.zeros((x_test.shape[0], self.num_samples))

        x_train_proj = x_train @ sampled_points

        x_test_proj = x_test @ sampled_points

        x_train_proj.sort(axis=0)
        for i in range(x_test.shape[0]):
            for j in range(self.num_samples):
                ranking[i, j] = np.argmin(
                    abs(x_train_proj[:, j] - x_test_proj[i, j]),
                )

        d_irw = ranking / len(self.x_train)

        for i, point in enumerate(d_irw):
            for j, sample in enumerate(point):
                d_irw[i, j] = min(sample, 1 - sample)

        d_irw = np.mean(d_irw, axis=1)

        return d_irw
