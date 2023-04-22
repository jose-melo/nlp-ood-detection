import numpy as np
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
from typing import List, Dict


class DataPreprocessing:
    dataset2key = {
        "imdb": "text",
        "sst2": "sentence",
        "trec": "text",
        "argilla/news-summary": "text",
        "race": "article",
        "yelp_review_full": "text",
        "paws": "sentence1",
    }

    dataset2args = {
        "imdb": {},
        "sst2": {},
        "trec": {},
        "argilla/news-summary": {},
        "race": {"name": "all"},
        "yelp_review_full": {},
        "paws": {"name": "labeled_final"},
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_list: List[str] = ["imdb"],
    ):
        self.dataset_list: list[str] = dataset_list
        self.datasets: dict[str, dict[str, Dataset]] = {}
        self.tokenizer = tokenizer

        for dataset in self.dataset_list:
            self.datasets[dataset] = self.__load_data(
                dataset,
                **DataPreprocessing.dataset2args[dataset],
            )

    def __preprocess(self, dataset: Dict[str, Dataset], dataset_name: str) -> Dataset:
        print("Dataset preprocessing...")

        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                lambda x: self.tokenizer(
                    x[DataPreprocessing.dataset2key[dataset_name]],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ),
                batched=True,
            )
            dataset[split].set_format(
                type="torch",
                columns=["input_ids", "attention_mask"],
            )
        return dataset

    def __load_data(self, dataset_name: str, **load_args) -> Dataset:
        print(f"Loading {dataset_name} dataset...")
        dataset = load_dataset(dataset_name, **load_args)

        split_train_val = np.arange(len(dataset["train"]))
        generator = np.random.default_rng()
        split_train_val = generator.permutation(split_train_val)
        train_split, val_split = train_test_split(
            split_train_val,
            test_size=0.2,
            random_state=42,
        )

        dataset = {
            "train": dataset["train"].select(train_split),
            "val": dataset["train"].select(val_split),
            "test": dataset["test"],
        }
        dataset = self.__preprocess(dataset, dataset_name)

        return dataset

    def __getitem__(self, dataset: str):
        return self.datasets[dataset]

    def __len__(self):
        return len(self.dataset_list)

    def __repr__(self):
        return f"DataLoader({self.dataset_list})"
