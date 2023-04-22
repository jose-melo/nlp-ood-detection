import numpy as np
import torch

from nlp_ood_detection.data_depth.utils import get_method
from nlp_ood_detection.data_processing.generate_data import LatentRepresentation


class NLPOODDetector:
    """Main class for the NLP OOD detector"""

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
        """Fit the model. Creates and prepares the data for the model"""
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
