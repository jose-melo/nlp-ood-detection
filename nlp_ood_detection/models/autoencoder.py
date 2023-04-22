import numpy as np
import pytorch_lightning as pl
import torch


class AutoEncoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=2):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden1 = torch.nn.Linear(hidden_dim, latent_dim)

        self.hidden2 = torch.nn.Linear(latent_dim, hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim)

        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.decoder(x))
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.loss(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.loss(x_hat, x)
        self.log("test_loss", loss)
        return loss


def compute_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    for batch in dataloader:
        embeddings.append(model.hidden1(model.encoder(batch)).detach().numpy())
    return np.concatenate(embeddings)
