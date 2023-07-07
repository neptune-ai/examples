import torch
import torch.nn as nn
from lightning import LightningModule


class LSTMRegressor(LightningModule):
    def __init__(
        self,
        n_features,
        hidden_dim,
        n_layers,
        criterion,
        dropout,
        learning_rate,
        seq_len,
        batch_first=True,
    ):
        super().__init__()

        self.save_hyperparameters()

        # loss
        self.criterion = criterion

        # lr
        self.learning_rate = learning_rate

        # n_features
        self.n_features = n_features

        # n_layers
        self.n_layers = n_layers

        # hidden_dim
        self.hidden_dim = hidden_dim

        # Model
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=batch_first,
            dropout=dropout,
        )
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.regressor(output[:, -1, :])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        return self._batch_step(batch, loss_name="train_loss")

    def validation_step(self, batch, batch_idx):
        return self._batch_step(batch, loss_name="val_loss")

    def _batch_step(self, batch, loss_name):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log(loss_name, loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def predict(self, test_loader):
        with torch.no_grad():
            x_test, y_test = next(iter(test_loader))
            self.eval()
            yhat = self(x_test)

        return yhat.detach().data.numpy(), y_test.detach().data.numpy()
