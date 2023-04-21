import numpy as np
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
from typing import List, Dict


class DataPreprocessing:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_list: List[str] = ["imdb"],
    ):
        self.dataset_list: list[str] = dataset_list
        self.datasets: dict[str, dict[str, Dataset]] = {}
        self.tokenizer = tokenizer

        for dataset in self.dataset_list:
            self.datasets[dataset] = self.__load_data(dataset)

    def __preprocess(self, dataset: Dict[str, Dataset]) -> Dataset:
        print("Dataset preprocessing...")

        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                lambda x: self.tokenizer(
                    x["text"],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ),
                batched=True,
            )
            dataset[split].set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
            )
        return dataset

    def __load_data(self, dataset: str) -> Dataset:
        print(f"Loading {dataset} dataset...")
        dataset = load_dataset(dataset)

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
        dataset = self.__preprocess(dataset)

        return dataset

    def __getitem__(self, dataset: str):
        return self.datasets[dataset]

    def __len__(self):
        return len(self.dataset_list)

    def __repr__(self):
        return f"DataLoader({self.dataset_list})"
