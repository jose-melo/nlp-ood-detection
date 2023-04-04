from abc import ABC, abstractmethod
from typing import Dict, Union, Optional
from numpy import ndarray
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import softmax
import numpy as np


class SimilarityScorerBase(ABC):
    """ Base class for similarity scorers"""

    def __init__(self,
                 data_train: ndarray,
                 model: Optional[Module] = None,
                 **kwargs) -> None:
        super(SimilarityScorerBase, self).__init__()
        self.data_train = data_train
        self.model = model
        self.x_test = None

    @abstractmethod
    def score(self, x: Union[Dict[str, Tensor], ndarray]) -> ndarray:
        """Calculate the similarity score
            :params: X (Tensor): input value
            :return: ndarray: the similarity score
        """


class Mahalanobis(SimilarityScorerBase):
    """ Mahalanobis distance based similarity scorer"""

    def __init__(self,
                 data_train: ndarray,
                 model: Optional[Module] = None,
                 **kwargs) -> None:
        super(Mahalanobis, self).__init__(data_train, model)

    def score(self, x: Union[Dict[str, Tensor], ndarray]) -> ndarray:
        self.x_test = x - np.mean(self.data_train, axis=0)
        cov = np.cov(self.data_train, rowvar=False)
        inv_cov = np.linalg.inv(cov)
        mahalanobis = np.dot(self.x_test, inv_cov)
        mahalanobis = np.dot(mahalanobis, self.x_test.T)

        return mahalanobis


class MSP(SimilarityScorerBase):
    """Implementation of the Maximum similarity probability (MSP) measure"""

    def __init__(self,
                 data_train: ndarray,
                 model: Optional[Module] = None,
                 **kwargs) -> None:
        super(MSP, self).__init__(data_train, model)

    def score(self, x: Union[Dict[str, Tensor], ndarray]) -> ndarray:
        self.x_test = x

        with torch.no_grad():
            logits = self.model(**self.x_test).cpu().numpy()
            prediction = softmax(logits, dim=1).argmax(dim=1)
            msp = 1 - np.maximum(prediction)

        return msp


class EnergyBased(SimilarityScorerBase):
    """Implementation of the Energy-based measure"""

    def __init__(self,
                 data_train: ndarray,
                 model: Optional[Module] = None,
                 temperature: float = 1,
                 **kwargs) -> None:
        super(EnergyBased, self).__init__(data_train, model)
        self.temperature = temperature

    def __lse(self, x: ndarray) -> ndarray:
        """ Implementation of the log-sum-exp function"""
        return -self.temperature*np.log(np.sum(np.exp(x)))

    def score(self, x: Union[Dict[str, Tensor], ndarray]) -> ndarray:
        self.x_test = x

        with torch.no_grad():
            logits = self.model(**self.x_test).cpu().numpy()
            prediction = softmax(logits, dim=1).argmax(dim=1)
            energy = self.__lse(prediction)

        return energy


class IRW(SimilarityScorerBase):
    """Implementation of the Integrated ranked-weigthed (IRW) measure"""

    def __init__(self,
                 data_train: ndarray,
                 model: Optional[Module] = None,
                 num_dim: int = 2,
                 num_samples: int = 1000,
                 **kwargs):
        super(IRW, self).__init__(data_train, model)
        self.num_dim = num_dim
        self.num_samples = num_samples

    def __sample_sphere(self):
        """Sample from the unit sphere"""
        u = np.random.randn(self.num_dim, self.num_samples)
        u /= np.linalg.norm(u, axis=0)
        return u

    def score(self, x: Union[Tensor, ndarray]) -> ndarray:
        self.x_test = x

        # Sample from the unitr sphere (Let's Monte Carlo it)
        sampled_points = self.__sample_sphere()

        ranking = np.zeros((self.x_test.shape[0], self.num_samples))

        # Projection of the train data on the sampled points  from the unit sphere
        data_train_proj = self.data_train @ sampled_points

        # Projection of the test data on the sampled points  from the unit sphere
        x_test_proj = self.x_test @ sampled_points

        # Calculate the ranking of the test data
        data_train_proj.sort(axis=0)
        for i in range(self.x_test.shape[0]):
            for j in range(self.num_samples):
                ranking[i, j] = np.argmin(
                    abs(data_train_proj[:, j] - x_test_proj[i, j]))

        # Normalize the ranking
        d_irw = ranking / len(self.data_train)

        # ID measure
        for i, point in enumerate(d_irw):
            for j, sample in enumerate(point):
                d_irw[i, j] = min(sample, 1 - sample)

        d_irw = np.mean(d_irw, axis=1)

        return d_irw
