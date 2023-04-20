import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from torch import Tensor

from nlp_ood_detection.data_depth.utils import get_method, load_data


class BasicClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, model: ClassifierMixin, **kwargs):
        self.model = model(**kwargs)

    def __call__(self, x_train: ndarray, y_train: ndarray, x_test: ndarray) -> Tensor:
        self.fit(x_train, y_train)
        proba = self.predict_proba(x_test)
        logits = np.log(proba) - np.log(1 - proba)
        return Tensor(logits).reshape(-1, 2)

    def score(self, x_test: ndarray, y_test: ndarray) -> float:
        return self.model.score(x_test, y_test)

    def fit(self, x_train: ndarray, y_train: ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict_proba(self, x_test: ndarray) -> ndarray:
        return self.model.predict_proba(x_test)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the similarity measures")

    parser.add_argument(r"--grid_size", help=r"Grid size", default=100, type=int)
    method_parser = parser.add_subparsers(
        dest=r"method",
        help="Chosen method",
        required=True,
    )

    # Parse the arguments for the energy method
    energy_parser = method_parser.add_parser(name=r"energy", help="Energy method")

    energy_parser.add_argument(
        r"--temperature",
        dest=r"T",
        help=r"Temperature",
        default=1,
        type=float,
    )

    # Parse the arguments for the IRW method
    irw_parser = method_parser.add_parser(name=r"irw", help="IRW method")
    irw_parser.add_argument(
        "--n_samples",
        help="Number of samples",
        default=1000,
        type=int,
    )
    irw_parser.add_argument("--n_dim", help="Number of dimensions", default=2, type=int)

    args, _ = parser.parse_known_args()
    args = vars(args)

    x_train, y_train, x_grid, xx, yy = load_data(grid_size=args["grid_size"])

    args = {
        **args,
        "use_logits": False,
        "x_train": x_train,
        "y_train": y_train,
    }

    model_params = {"model": SVC, "probability": True}
    args["model"] = BasicClassifier(**model_params)

    scorer = get_method(**args)

    score = scorer.score(x_grid)

    plt.contourf(xx, yy, score.reshape(xx.shape), levels=20, cmap="viridis")
    plt.scatter(
        x_train[:, 0],
        x_train[:, 1],
        label="Train data",
        marker="x",
        c="r",
        s=50,
    )
    plt.colorbar()
    plt.title(args["method"])
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.show()


if __name__ == "__main__":
    main()
