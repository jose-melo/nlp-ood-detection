import argparse
from matplotlib import pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from nlp_ood_detection.data.bert_datamodule import BertBasedDataModule


def main():
    parser = argparse.ArgumentParser(description="Run a script")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="The script to run",
        default=["imdb"],
    )

    parser.add_argument(
        "--model_name",
        help="The model to use",
        default="textattack/distilbert-base-uncased-imdb",
        type=str,
    )

    args, _ = parser.parse_known_args()

    #datasets_names = args.datasets
    model_name = args.model_name

    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    bert = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    params = {
        "model_name": model_name,
        "config": config,
        "max_epochs": 10,
        "lr": 1e-5,
        "batch_size": 8,
        "num_workers": 0,
        "tokenizer": tokenizer,
        "dataset_name": "imdb",
        "num_labels": 2,
    }
    dataloader = BertBasedDataModule(**params)

    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()

    latent = []
    for idx, batch in enumerate(tqdm(train_dataloader,
                                     desc="Latent calculation",
                                     total=len(train_dataloader))):

        outputs = bert(batch["input_ids"], batch["attention_mask"])
        latent_representation = np.array(
            [x[:, -1, :].detach().numpy() for x in outputs["hidden_states"]])
        mean_latent_representation = np.mean(latent_representation, axis=0)
        latent.append(mean_latent_representation)
        if idx == 2:
            break
    latent = np.concatenate(latent, axis=0)

    pca = PCA(n_components=2)
    transformed = pca.fit_transform(mean_latent_representation)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.scatter(transformed[:, 0], transformed[:, 1],
                c=batch["label"], cmap="tab10")
    plt.title(f"PCA: Latent representation - {model_name} dataset")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()


if __name__ == '__main__':
    main()
