import argparse
import copy

import numpy as np
import torch
from numpy import ndarray
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from nlp_ood_detection.data_depth.utils import get_method
from nlp_ood_detection.data_processing.generate_data import LatentRepresentation


def main():
    parser = argparse.ArgumentParser(description="Test the similarity measures")

    parser.add_argument(r"--run_all", help=r"Run all methods", action="store_true")
    parser.add_argument(r"--grid_size", help=r"Grid size", default=100, type=int)
    parser.add_argument(
        r"--datasets",
        nargs="+",
        help="The script to run",
        default=["imdb"],
    )

    parser.add_argument(
        r"--aggregations",
        nargs="+",
        help="A list of aggregations to use",
        default=["mean"],
    )

    parser.add_argument(
        r"--model_name",
        help="The model to use",
        default="textattack/distilbert-base-uncased-imdb",
        type=str,
    )

    parser.add_argument(
        r"--data_folder",
        help="The data folder",
        default="data",
        type=str,
    )

    method_parser = parser.add_subparsers(
        dest=r"method",
        help="Chosen method",
    )

    # Parse the arguments for the mahalanobis method
    energy_parser = method_parser.add_parser(name=r"maha", help="Mahalanobis method")

    # Parse the arguments for the msp method
    energy_parser = method_parser.add_parser(name=r"msp", help="MSP method")

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

    scores = generate_scores(**args)
    print(scores)


class NLPOODDetector:
    def __init__(
        self,
        dataset_in: str,
        dataset_out: str,
        aggregation: str,
        data_folder: str,
        method: str,
        model_name: str,
        max_size: int = 1000,
        threshold: float = 0.5,
        **kwargs,
    ):
        self.model_name = model_name
        self.data_folder = data_folder
        self.dataset_in = dataset_in
        self.dataset_out = dataset_out
        self.aggregation = aggregation
        self.method = method
        self.max_size = max_size
        self.threshold = threshold

    def fit(self, **kwargs):
        data = LatentRepresentation.load(
            dataset_names=[self.dataset_in, self.dataset_out],
            aggregations=[self.aggregation],
            output_folder=self.data_folder,
            model_name=self.model_name,
        )

        x_train = data[self.dataset_in][self.aggregation]["hidden_states"]
        y_train = data[self.dataset_in][self.aggregation]["label"]
        x_test = data[self.dataset_out][self.aggregation]["hidden_states"]
        y_test = data[self.dataset_out][self.aggregation]["label"]
        logits = data[self.dataset_out][self.aggregation]["logits"]

        del data

        generator = np.random.default_rng()
        if len(x_train) > self.max_size:
            idx = generator.choice(len(x_train), size=self.max_size, replace=False)
            x_train = x_train[idx]
            y_train = y_train[idx]

        if len(x_test) > self.max_size:
            idx = generator.choice(len(x_test), size=self.max_size, replace=False)
            x_test = x_test[idx]
            y_test = y_test[idx]
            logits = logits[idx]

        self.params = {
            "x_train": x_train,
            "y_train": y_train,
            "x": x_test,
            "labels": None,
            "num_dim": x_test.shape[1],
            "num_samples": 1000,
            "n_dirs": 1000,
            "logits": torch.Tensor(logits),
            "feature": list(range(x_test.shape[1])),
            **kwargs,
        }

    def score(self, **kwargs) -> float:
        method = get_method(self.method, **self.params)
        return method.score(**self.params)

    def predict(self, **kwargs) -> float:
        method = get_method(self.method, **self.params)
        score = method.score(**self.params)
        return (score > self.threshold).astype(int)


def generate_scores(
    datasets: list[tuple[str, str]],
    aggregations: list[str],
    data_folder: str,
    methods: list[str],
    model_name: str,
    dataset_in: str,
    **kwargs,
) -> ndarray[float]:
    dataset_names = list(
        {dataset for dataset_pair in datasets for dataset in dataset_pair},
    )
    data = LatentRepresentation.load(
        dataset_names=dataset_names,
        aggregations=aggregations,
        output_folder=data_folder,
        model_name=model_name,
    )

    scores = {}
    for a, b in datasets:
        if a not in scores:
            scores[a] = {}
        scores[a][b] = {
            method: {aggregation: None for aggregation in aggregations}
            for method in methods
        }
    metrics = copy.deepcopy(scores)

    for method in methods:
        for train_data, eval_data in datasets:
            for aggregation in aggregations:
                x_train = data[train_data][aggregation]["hidden_states"]
                y_train = data[train_data][aggregation]["label"]
                x_test = data[eval_data][aggregation]["hidden_states"]
                y_test = data[eval_data][aggregation]["label"]
                logits = data[eval_data][aggregation]["logits"]

                # sample the data
                generator = np.random.default_rng()

                if len(x_train) > 1000:
                    idx = generator.choice(len(x_train), size=1000, replace=False)
                    x_train = x_train[idx]
                    y_train = y_train[idx]

                if len(x_test) > 1000:
                    idx = generator.choice(len(x_test), size=1000, replace=False)
                    x_test = x_test[idx]
                    y_test = y_test[idx]
                    logits = logits[idx]

                params = {
                    "x_train": x_train,
                    "y_train": y_train,
                    "x": x_test,
                    "labels": None,
                    "num_dim": x_test.shape[1],
                    "num_samples": 1000,
                    "n_dirs": 1000,
                    "logits": torch.Tensor(logits),
                    "feature": list(range(x_test.shape[1])),
                    **kwargs,
                }
                scorer = get_method(method, **params)

                score = scorer.score(**params)
                scores[train_data][eval_data][method][aggregation] = score

    for train_data, eval_data in datasets:
        for aggregation in aggregations:
            for method in methods:
                score_in = scores[train_data][dataset_in][method][aggregation]
                score_out = scores[train_data][eval_data][method][aggregation]

                score = np.concatenate([score_in, score_out])
                y_test = np.concatenate(
                    [np.zeros_like(score_in), np.ones_like(score_out)],
                )

                fpr, tpr, thresholds = roc_curve(y_test, score)
                metrics[train_data][eval_data][method][aggregation] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresholds,
                    "auc_roc": roc_auc_score(y_test, score),
                    "auc_pr": average_precision_score(y_test, score),
                    "fpr@95": fpr[np.argmin(np.abs(tpr - 0.95))],
                }

    return metrics, scores


if __name__ == "__main__":
    main()
