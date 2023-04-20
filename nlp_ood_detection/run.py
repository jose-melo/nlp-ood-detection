import argparse

from data_processing.data_processing import DataPreprocessing
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a script")
    parser.add_argument("--dataset", help="The script to run", default="imdb", type=str)
    parser.add_argument(
        "--model",
        help="The model to use",
        default="distilbert-base-cased",
        type=str,
    )
    args, _ = parser.parse_known_args()

    dataset = args.dataset
    model = args.model

    dataloader = DataPreprocessing(
        tokenizer=AutoTokenizer.from_pretrained("distilbert-base-cased"),
        dataset_list=["imdb"],
    )
