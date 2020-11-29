from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from lightning import LitModel

from torch.utils.data import DataLoader, random_split
from dataloaders import TensorDispatcher, AugmentedDataset


def csv_to_tensor(filename):
    df = pd.read_csv(filename)
    arr = np.nan_to_num(df.to_numpy())
    norm_tensor = 0.5 + torch.from_numpy(arr) / 2 ** 16
    norm_tensor = norm_tensor.transpose(0, 1)
    return norm_tensor.float()


class Volcano1DDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        data_dir,
        batch_size,
        train_split=0.9,
        augmenter=None,
        maxmem=5e9,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_split = train_split
        self.data_dir = data_dir
        self.train_df = train_df
        self.augmenter = augmenter
        self.maxmem = maxmem

    def setup(self, stage=None):
        volcano_data_full = Greedy1DTensorLoader(
            self.train_df, self.data_dir, self.maxmem
        )
        db_size = len(volcano_data_full)
        train_size = int(db_size * self.train_split)
        val_size = db_size - train_size
        volcano_train, self.volcano_val = random_split(
            volcano_data_full, [train_size, val_size]
        )

        self.volcano_train = AugmentedDataset(volcano_train, self.augmenter)

    def train_dataloader(self):
        return DataLoader(self.volcano_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.volcano_val, batch_size=self.batch_size)


class Greedy1DTensorLoader(TensorDispatcher):
    def __init__(self, train_df, data_dir, maxmem=5e9, augmenter=None):
        print("Setting up greedy tensor loader")
        my_iter = zip(train_df["segment_id"], train_df["time_to_eruption"])
        spectra = []
        responses = []

        mem = 0
        for i, (x, y) in enumerate(my_iter):
            filepath = Path(data_dir) / f"{x}.csv"
            data_tensor = csv_to_tensor(filepath).half()
            spectra.append(data_tensor)

            y = [float(y) / 1e8]
            responses.append(torch.tensor(y))
            mem += data_tensor.element_size() * data_tensor.nelement()

            if mem > maxmem:
                print(f"Maximum memmory reached, read {i} tensors")
                break

            if i % 100 == 0:
                print(f"Read {i} files so far...")

        print(f"Loaded dataset uses aprox {mem} bytes, {mem/1000000} MB")

        super(Greedy1DTensorLoader, self).__init__(
            data_tensor=torch.stack(spectra),
            response_tensor=torch.stack(responses),
            augmenter=augmenter,
        )

    def __len__(self):
        return super(Greedy1DTensorLoader, self).__len__()

    def __getitem__(self, index):
        return super(Greedy1DTensorLoader, self).__getitem__(index)


class ConvNet1D(pl.LightningModule):
    def __init__(self):
        super(ConvNet1D, self).__init__()
        self.conv1 = torch.nn.Conv1d(10, 32, 9, stride=5)
        self.pool = torch.nn.MaxPool1d(2, 2)
        self.conv2 = torch.nn.Conv1d(32, 64, 9, stride=5)
        self.fc1 = torch.nn.Linear(64 * 599, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*599)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.leaky_relu(x)
        return x


class LitConvNet1D(LitModel):
    def __init__(self, *args, **kwargs):
        net = ConvNet1D()
        super(LitConvNet1D, self).__init__(net, *args, **kwargs)


if __name__ == "__main__":
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

    lr = 1e-2
    opt = torch.optim.Adam
    model_name = "LitConvNet1D"

    full_df = pd.read_csv("./sample_data/train.csv")
    data_dir = "../train/"

    data = Volcano1DDataModule(full_df, data_dir, augmenter=None, batch_size=64, maxmem=1e8)

    optim_name = opt.__name__
    run_name = f"{model_name}_{optim_name}_{lr}"
    print(f">>>>>>>>>>>>>> {run_name}")
    if not "Stopping Enabled":
        stopper = EarlyStopping(monitor="val_loss", verbose=True, patience=20, mode="min")
        callbacks = [stopper]
    else:
        callbacks = []


    if not "Logging Enabled":
        logger = TensorBoardLogger("tb_logs", name=run_name)
        wandb_logger = WandbLogger(name=run_name, project="volcanos")

        loggers = [logger, wandb_logger]
    else:
        loggers = []

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        auto_lr_find=False,
        gpus=0,
        # precision=16,
        progress_bar_refresh_rate=10,
        max_epochs=1000,
        profiler="simple",
    )

    model = LitConvNet1D(learning_rate=lr, optimizer=opt)
    # trainer.tune(model, data)
    trainer.fit(model, data)
