import argparse

import numpy as np
import pandas as pd
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
        dataset_name: str,
        output_folder: str = ".",
        aggregations: list[str] = ["mean"],
    ):
        self.model_name = model_name
        self.aggregations = aggregations
        config = AutoConfig.from_pretrained(self.model_name, output_hidden_states=True)
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config,
        )
        self.output_folder = output_folder
        self.dataset_name = dataset_name

    def __call__(self, batch) -> dict[str, np.ndarray]:
        outputs = self.bert(batch["input_ids"], batch["attention_mask"])

        agg = {}
        for aggregation in self.aggregations:
            if aggregation == "mean":
                latent_representation = np.array(
                    [x[:, -1, :].detach().numpy() for x in outputs["hidden_states"]],
                )
                latent = np.mean(latent_representation, axis=0)

            if aggregation == "last":
                latent = outputs["hidden_states"][-1][:, -1, :].detach().numpy()

            if aggregation == "two_last":
                latent = np.concatenate(
                    [
                        outputs["hidden_states"][-1][:, -1, :].detach().numpy(),
                        outputs["hidden_states"][-2][:, -1, :].detach().numpy(),
                    ],
                    axis=1,
                )

            agg[aggregation] = latent
            filename = f"latent_{self.dataset_name}_{aggregation}.parquet"
            latent = np.concatenate(
                [latent, batch["label"].detach().numpy().reshape(-1, 1)],
                axis=1,
            )
            self._save(filename, latent)

        return agg

    def _save(self, filename: str, latent: np.ndarray):
        df = pd.DataFrame(latent)

        # Set the columns
        df.columns = [str(i) for i in range(latent.shape[1] - 1)] + ["label"]

        # set the types
        df.label = df.label.astype(int)

        parquet_params = {"compression": "gzip", "index": False, "engine": "pyarrow"}
        filename = f"{self.output_folder}/{filename}.parquet"
        df.to_parquet(filename, **parquet_params)

    def load(self, path):
        df = pd.read_parquet(path)
        return df


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


def generate_latent_data(datasets_names, model_name, aggregations):
    for dataset_name in datasets_names:
        print(f"Dataset: {dataset_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        params = {
            "model_name": model_name,
            "max_epochs": 10,
            "lr": 1e-5,
            "batch_size": 8,
            "num_workers": 0,
            "tokenizer": tokenizer,
            "dataset_name": dataset_name,
            "num_labels": 2,
        }
        dataloader = BertBasedDataModule(**params)
        dataloader.setup()
        train_dataloader = dataloader.train_dataloader()

        latent_generator = LatentRepresentation(
            model_name=model_name,
            dataset_name=dataset_name,
            output_folder="data",
            aggregations=aggregations,
        )

        latent_data = {aggregation: [] for aggregation in aggregations}
        for idx, batch in enumerate(
            tqdm(
                train_dataloader,
                desc="Latent calculation",
                total=len(train_dataloader),
            ),
        ):
            latent = latent_generator(batch)
            for aggregation in aggregations:
                latent_data[aggregation].append(latent[aggregation])
            if idx == 2:
                break

        for aggregation in aggregations:
            latent_data[aggregation] = np.concatenate(
                latent_data[aggregation],
                axis=0,
            )

    return latent_data


if __name__ == "__main__":
    main()
