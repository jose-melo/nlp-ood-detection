import argparse
import copy

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from nlp_ood_detection.data_processing.bert_datamodule import BertBasedDataModule


class LatentRepresentation:
    def __init__(
        self,
        model_name: str,
        dataset_names: list[str],
        output_folder: str = ".",
        aggregations: list[str] = ["mean"],
        n_classes: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.aggregations = aggregations
        self.device = device
        config = AutoConfig.from_pretrained(self.model_name, output_hidden_states=True)
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config,
        )
        self.bert.to(self.device)
        self.bert.eval()
        self.output_folder = output_folder
        self.dataset_names = dataset_names
        self.n_classes = n_classes

    def __call__(self, batch) -> dict[str, np.ndarray]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        outputs = self.bert(input_ids, attention_mask)

        agg = {
            aggregation: {"hidden_states": [], "logits": [], "label": []}
            for aggregation in self.aggregations
        }
        for aggregation in self.aggregations:
            if aggregation == "mean":
                latent_representation = np.array(
                    [
                        x[:, -1, :].detach().cpu().numpy()
                        for x in outputs["hidden_states"]
                    ],
                )
                latent = np.mean(latent_representation, axis=0)

            if aggregation == "last":
                latent = outputs["hidden_states"][-1][:, -1, :].detach().cpu().numpy()

            if aggregation == "two_last":
                latent = np.concatenate(
                    [
                        outputs["hidden_states"][-1][:, -1, :].detach().cpu().numpy(),
                        outputs["hidden_states"][-2][:, -1, :].detach().cpu().numpy(),
                    ],
                    axis=1,
                )

            logits = outputs["logits"].detach().cpu().numpy()
            label = batch["label"].detach().cpu().numpy()
            agg[aggregation]["hidden_states"] = latent
            agg[aggregation]["logits"] = logits
            agg[aggregation]["label"] = label

        return agg

    def _save(self, data: dict[str, dict[str, dict[str, np.ndarray]]]):
        model_name = self.model_name.split("/")[-1]
        for dataset_name in self.dataset_names:
            for aggregation in self.aggregations:
                if len(data[dataset_name][aggregation]["hidden_states"]) == 0:
                    continue
                filename = f"latent_{model_name}_{dataset_name}_{aggregation}"
                latent = data[dataset_name][aggregation]["hidden_states"]
                logits = data[dataset_name][aggregation]["logits"]
                labels = data[dataset_name][aggregation]["label"].reshape(-1, 1)
                data_to_save = np.concatenate(
                    [
                        latent,
                        logits,
                        labels,
                    ],
                    axis=1,
                )
                columns = (
                    [str(i) for i in range(latent.shape[1])]
                    + [f"logit_{i}" for i in range(logits.shape[1])]
                    + ["label"]
                )
                df = pd.DataFrame(data_to_save)
                df.columns = columns

                parquet_params = {
                    "compression": "gzip",
                    "index": False,
                    "engine": "pyarrow",
                }
                filename = f"{self.output_folder}/{filename}.parquet"
                df.to_parquet(filename, **parquet_params)

    @staticmethod
    def load(
        dataset_names: str,
        aggregations: str,
        output_folder: str,
        model_name: str,
    ):
        data = {
            dataset_name: {
                aggregation: {"logits": None, "hidden_states": None, "label": None}
                for aggregation in aggregations
            }
            for dataset_name in dataset_names
        }
        for dataset_name in dataset_names:
            for aggregation in aggregations:
                model_name = model_name.split("/")[-1]
                filename = f"latent_{model_name}_{dataset_name}_{aggregation}.parquet"
                path = f"{output_folder}/{filename}"
                df = pd.read_parquet(path)
                data[dataset_name][aggregation]["logits"] = df.filter(
                    regex="logit",
                ).values
                data[dataset_name][aggregation]["hidden_states"] = df.filter(
                    regex="^\\d+$",
                ).values
                data[dataset_name][aggregation]["label"] = df.label.values
        return data


def main():
    parser = argparse.ArgumentParser(description="Run a script")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="The script to run",
        default=["imdb"],
    )

    parser.add_argument(
        "--aggregations",
        nargs="+",
        help="A list of aggregations to use",
        default=["mean"],
    )

    parser.add_argument(
        "--model_name",
        help="The model to use",
        default="textattack/distilbert-base-uncased-imdb",
        type=str,
    )

    args, _ = parser.parse_known_args()

    datasets_names = args.datasets
    model_name = args.model_name
    aggregations = args.aggregations

    generate_latent_data(datasets_names, model_name, aggregations)


def generate_latent_data(dataset_names, model_name, aggregations):
    latent_data = {
        dataset: {
            aggregation: {"hidden_states": [], "logits": [], "label": []}
            for aggregation in aggregations
        }
        for dataset in dataset_names
    }

    latent_generator = LatentRepresentation(
        model_name=model_name,
        dataset_names=dataset_names,
        output_folder="data",
        aggregations=aggregations,
    )

    for dataset_name in dataset_names:
        print(f"Dataset: {dataset_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        params = {
            "model_name": model_name,
            "max_epochs": 10,
            "lr": 1e-5,
            "batch_size": 4,
            "num_workers": 0,
            "tokenizer": tokenizer,
            "dataset_name": dataset_name,
            "num_labels": 2,
        }
        dataloader = BertBasedDataModule(**params)
        dataloader.setup()
        test_dataloader = dataloader.test_dataloader()

        for idx, batch in enumerate(
            tqdm(
                test_dataloader,
                desc="Latent calculation",
                total=len(test_dataloader),
            ),
        ):
            data = latent_generator(batch)
            for aggregation in aggregations:
                for key in latent_data[dataset_name][aggregation].keys():
                    latent_data[dataset_name][aggregation][key].append(
                        data[aggregation][key],
                    )
            if idx % 1000 == 0:
                backup = copy.deepcopy(latent_data)
                for aggregation in aggregations:
                    for key in backup[dataset_name][aggregation].keys():
                        backup[dataset_name][aggregation][key] = np.array(
                            np.concatenate(
                                backup[dataset_name][aggregation][key],
                                axis=0,
                            ),
                        )
                latent_generator._save(backup)

        for aggregation in aggregations:
            for key in latent_data[dataset_name][aggregation].keys():
                latent_data[dataset_name][aggregation][key] = np.concatenate(
                    latent_data[dataset_name][aggregation][key],
                    axis=0,
                )
    latent_generator._save(latent_data)

    return latent_data


if __name__ == "__main__":
    main()
