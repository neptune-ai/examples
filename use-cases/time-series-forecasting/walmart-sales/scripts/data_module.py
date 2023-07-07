import sys

import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

sys.path.append("../")
from utils import load_data


class TimeseriesDataset(Dataset):
    """
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    """

    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len - 1)

    def __getitem__(self, index):
        return (
            self.X[index : (index + self.seq_len)],
            self.y[index + self.seq_len - 1],
        )


class WalmartSalesDataModule(LightningDataModule):
    def __init__(self, seq_len=2, batch_size=64, year=2010, num_workers=0, path="train_data.csv"):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.year = year
        self.num_workers = num_workers
        self.path = path
        self.allow_zero_length_dataloader_with_multiple_devices = True

    def setup(self, stage):
        if stage == "fit" and self.X_train is not None:
            return
        if stage == "test" and self.X_valid is not None:
            return
        if stage is None and self.X_train is not None and self.X_valid is not None:
            return

    def prepare_data_per_node(self):
        pass

    def prepare_data(self):
        self.df = load_data(self.path, cache=True)

        (
            self.X_train,
            self.X_valid,
            self.X_test,
            self.y_train,
            self.y_valid,
            self.y_test,
        ) = self.get_train_valid_split(self.df[self.df.Year == self.year])

    def train_dataloader(self):
        train_loader, train_ds = self.create_dataloader(
            self.X_train, self.y_train, self.batch_size, seq_len=self.seq_len
        )
        return train_loader

    def val_dataloader(self):
        valid_loader, valid_ds = self.create_dataloader(
            self.X_valid, self.y_valid, self.batch_size, seq_len=self.seq_len
        )
        return valid_loader

    def test_dataloader(self):
        test_loader, test_ds = self.create_dataloader(
            self.X_test, self.y_test, self.batch_size, seq_len=self.seq_len
        )
        return test_loader

    def _log_hyperparams(self):
        pass

    def create_dataloader(self, X, y, bs, seq_len: int = 1):
        features = torch.Tensor(X)
        targets = torch.Tensor(y).float()
        features_ds = TimeseriesDataset(features, targets, seq_len=seq_len)
        data_loader = DataLoader(
            features_ds, batch_size=bs, shuffle=False, num_workers=self.num_workers
        )

        return data_loader, features_ds

    def get_train_valid_split(self, df):
        X = df["Weekly_Sales"].values.reshape((-1, 1))
        y = df["Weekly_Sales"].values.reshape((-1, 1))

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.30, random_state=42, shuffle=False
        )

        X_train, y_train, X_valid, y_valid = self.scaling_data(X_train, y_train, X_valid, y_valid)

        X_valid, X_test, y_valid, y_test = train_test_split(
            X_valid, y_valid, test_size=0.15, random_state=42, shuffle=False
        )

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def scaling_data(self, X_train, y_train, X_valid, y_valid):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = scaler.fit_transform(X_train)
        y_train = scaler.transform(y_train)
        X_valid = scaler.transform(X_valid)
        y_valid = scaler.transform(y_valid)
        self.scaler = scaler

        return X_train, y_train, X_valid, y_valid
