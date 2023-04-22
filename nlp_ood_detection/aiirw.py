# Taken from Guillaume Staerman
# Based on the paper: https://arxiv.org/abs/2106.11068


import numpy as np
from sklearn.covariance import MinCovDet as MCD
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from nlp_ood_detection.data_depth.similarity_scorer import SimilarityScorerBase

########################################################
#################### Some useful functions ########################
########################################################


def cov_matrix(X, robust=False):
    """Compute the covariance matrix of X."""
    if robust:
        cov = MCD().fit(X)
        sigma = cov.covariance_
    else:
        sigma = np.cov(X.T)

    return sigma


def standardize(X, robust=False):
    """Compute the square inverse of the covariance matrix of X."""

    sigma = cov_matrix(X, robust)
    n_samples, n_features = X.shape
    rank = np.linalg.matrix_rank(sigma)

    if rank < n_features:
        pca = PCA(rank)
        pca.fit(X)
        X_transf = pca.fit_transform(X)
        sigma = cov_matrix(X_transf)
    else:
        X_transf = X.copy()

    u, s, _ = np.linalg.svd(sigma)
    square_inv_matrix = u / np.sqrt(s)

    return X_transf @ square_inv_matrix, square_inv_matrix


########################################################
#################### Sampled distributions ########################
########################################################


def sampled_sphere(n_dirs, d):
    """Produce ndirs samples of d-dimensional uniform distribution on the
    unit sphere
    """

    mean = np.zeros(d)
    identity = np.identity(d)
    U = np.random.multivariate_normal(mean=mean, cov=identity, size=n_dirs)

    return normalize(U)


def Wishart_matrix(d):
    cov = np.random.randn(d, d)
    return cov.dot(cov.T)


def multivariate_t(mu, sigma, t, m):
    """
    Produce m samples of d-dimensional multivariate t distribution

    Args:
        mu (numpy.ndarray): mean vector
        sigma (numpy.ndarray): scale matrix (covariance)
        t (float): degrees of freedom
        m (int): # of samples to produce

    Returns:
        numpy.ndarray
    """
    d = len(mu)
    g = np.tile(np.random.gamma(t / 2, 2 / t, m), (d, 1)).T
    z = np.random.multivariate_normal(np.zeros((d)), sigma, m)
    return mu + z / np.sqrt(g)


class AIIRW(SimilarityScorerBase):
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray | None = None,
        model: None = None,
        num_dim: int = 2,
        num_samples: int = 100,
        feature: list | None = [0, 1],
        **kwargs,
    ):
        super().__init__(x_train, y_train, model, feature)

    def score(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return 1 - ai_irw(self.x_train, X_test=x, **kwargs)


def ai_irw(
    X,
    AI=True,
    robust=False,
    n_dirs=100,
    X_test=None,
    random_state=None,
    **kwargs,
):
    """Compute the score of the (Affine-invariant-) integrated rank
        weighted depth of X_test w.r.t. X

    Parameters
    ----------

    X: Array of shape (n_samples, n_features)
            The training set.

    AI: bool
        if True, the affine-invariant version of irw is computed.
        If False, the original irw is computed.

    robust: bool, default=False
        if robust is true, the MCD estimator of the covariance matrix
        is performed.

    n_dirs: int | None
        The number of random directions needed to approximate
        the integral over the unit sphere.
        If None, n_dirs is set as 100* n_features.

    X_test: Array of shape (n_samples_test, n_features)
        The testing set.
        If None, return the score of the training sample.

    random_state: int | None
        The random state.

    Returns
    -------
    ai_irw_score: Array
        Depth score of each element in X_test.
        If X_test is None, return the score of the training sample.
    """

    # Setting seed:
    if random_state is None:
        random_state = 0

    np.random.seed(random_state)

    if X_test is None:
        if AI:
            X_reduced, _ = standardize(X, robust)
        else:
            X_reduced = X.copy()

        n_samples, n_features = X_reduced.shape

        # Setting the number of directions to 100 times the number of features as in the paper.
        if n_dirs is None:
            n_dirs = n_features * 100

        # Simulated random directions on the unit sphere.
        U = sampled_sphere(n_dirs, n_features)

        sequence = np.arange(1, n_samples + 1)
        depth = np.zeros((n_samples, n_dirs))

        proj = np.matmul(X_reduced, U.T)
        rank_matrix = np.matrix.argsort(proj, axis=0)

        for k in range(n_dirs):
            depth[rank_matrix[:, k], k] = sequence

        depth = depth / (n_samples * 1.0)

        ai_irw_score = np.mean(np.minimum(depth, 1 - depth), axis=1)

    else:
        if AI:
            X_reduced, Sigma_inv_square = standardize(X, robust)
            X_test_reduced = X_test @ Sigma_inv_square
        else:
            X_reduced = X.copy()
            X_test_reduced = X_test.copy()

        n_samples, n_features = X_reduced.shape
        n_samples_test, _ = X_test_reduced.shape

        # Setting the number of directions to 100 times the number of features as in the paper.
        if n_dirs is None:
            n_dirs = n_features * 100

        # Simulated random directions on the unit sphere.
        U = sampled_sphere(n_dirs, n_features)

        proj = np.matmul(X_reduced, U.T)
        proj_test = np.matmul(X_test_reduced, U.T)

        sequence = np.arange(1, n_samples_test + 1)
        depth = np.zeros((n_samples_test, n_dirs))
        temp = np.zeros((n_samples_test, n_dirs))

        proj.sort(axis=0)
        for k in range(n_dirs):
            depth[:, k] = np.searchsorted(a=proj[:, k], v=proj_test[:, k], side="left")

        depth /= n_samples * 1.0

        ai_irw_score = np.mean(np.minimum(depth, 1 - depth), axis=1)

    return ai_irw_score
