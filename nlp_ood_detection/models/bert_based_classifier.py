import argparse
from datetime import datetime
from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.nn import CrossEntropyLoss, Dropout, Linear, ReLU
from torch import softmax, sum
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from nlp_ood_detection.data.bert_datamodule import BertBasedDataModule


class BertBasedClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "distilbert-base-cased",
        config: AutoConfig = None,
        max_epochs: int = 10,
        lr: float = 1e-5,
        num_labels: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.max_epochs = max_epochs
        self.lr = lr
        self.config = config
        self.num_labels = num_labels

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
        )

        print("config: ", self.config)
        self.dropout = Dropout(0.1)
        self.linear1 = Linear(self.config.hidden_size, self.config.hidden_size)
        self.relu = ReLU()
        self.linear2 = Linear(self.config.hidden_size, self.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        label: torch.Tensor = None,
    ):
        x = self.bert(input_ids, attention_mask, output_hidden_states=True)

        # TODO: check if this is correct
        x = self.dropout(x["hidden_states"][-1][:, -1, :])
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def configure_optimizers(self):
        adamw = torch.optim.AdamW(
            self.parameters(), self.lr, weight_decay=1e-3)

        lr_decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            adamw, self.max_epochs)

        return [adamw], [lr_decay]

    def on_train_epoch_end(self):
        self.log_dict(
            {"lr": self.lr_schedulers().get_last_lr()[
                0], "epoch": self.current_epoch},
        )

    def calculate_loss(self, batch: Dict[str, torch.Tensor], mode):
        batch = {key: value.to(self.device) for key, value in batch.items()}

        labels = batch["label"]
        predict = self.forward(**batch)
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(predict, labels)

        prediction = softmax(predict, dim=1).argmax(dim=1)
        m = {}
        m["tp"] = ((prediction == 1) & (labels == 1)).float().sum()
        m["fp"] = ((prediction == 1) & (labels == 0)).float().sum()
        m["tn"] = ((prediction == 0) & (labels == 0)).float().sum()
        m["fn"] = ((prediction == 0) & (labels == 1)).float().sum()

        self.log("%s_loss" % mode, loss)
        for metric, value in m.items():
            self.log_dict(
                {
                    "ext/" + metric + "_" + mode: float(value),
                    r"step": float(self.current_epoch),
                },
                on_step=False,
                on_epoch=True,
                reduce_fx=sum,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.calculate_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.calculate_loss(batch, "test")


if __name__ == "__main__":
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
        default="distilbert-base-cased",
        type=str,
    )

    args, _ = parser.parse_known_args()

    datasets_names = args.datasets
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    params = {
        "model_name": model_name,
        "config": config,
        "max_epochs": 10,
        "lr": 1e-5,
        "batch_size": 5,
        "num_workers": 0,
        "tokenizer": tokenizer,
        "dataset_name": "imdb",
        "num_labels": 2,
    }

    dataloader = BertBasedDataModule(**params)

    model = BertBasedClassifier(**params)

    callback = ModelCheckpoint(monitor=r"val_loss", mode="min")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=params["max_epochs"],
        log_every_n_steps=1,
        callbacks=[callback],
    )

    trainer.fit(model, dataloader)

    trainer.save_checkpoint(
        f'ckpt_save_{datetime.now().strftime("%Y%m%d%H%M%S")}.ckpt')
