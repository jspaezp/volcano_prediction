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
        self.pool = torch.nn.MaxPool1d(4, 4)
        self.conv1 = torch.nn.Conv1d(10, 32, 9, stride=5)
        self.conv2 = torch.nn.Conv1d(32, 64, 9, stride=5)
        self.conv3 = torch.nn.Conv1d(64, 124, 9, stride=5)
        self.fc1 = torch.nn.Linear(124 * 74, 512)
        self.fc2 = torch.nn.Linear(512, 124)
        self.fc3 = torch.nn.Linear(124, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        print(x.shape)
        x = x.view(-1, 124 * 74)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.leaky_relu(x)
        return x


def conv9x1(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> torch.nn.Conv1d:
    """9x1 convolution with padding"""
    return torch.nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=9,
        stride=stride,
        padding=dilation + 3, # TODO find a better way to define this ...
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv1d:
    """1x1 convolution"""
    return torch.nn.Conv1d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock1D(torch.nn.Module):
    def __init__(self, inplanes, planes, stride, downsample = None):
        super().__init__()
        self.conv1 = conv9x1(inplanes, planes, stride)
        self.bn1 = torch.nn.BatchNorm1d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv9x1(planes, planes)
        self.bn2 = torch.nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        print(x.shape)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        print(out.shape)

        if self.downsample is not None:
            identity = self.downsample(x)

        print(identity.shape)
        out += identity
        out = self.relu(out)
        return out


class ResNet1D(pl.LightningModule):
    def __init__(self, out_features = 1):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(10, 64, 49, 4, 3)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.maxpool = torch.nn.MaxPool1d(9, 4)
        self.layer1 = torch.nn.Sequential(
            BasicBlock1D(64, 64, 1),
            BasicBlock1D(64, 64, 1)
        )
        self.layer2 = torch.nn.Sequential(
            BasicBlock1D(
                64, 128, 2, 
                downsample=torch.nn.Sequential(
                    conv1x1(64, 128, stride = 2),
                    torch.nn.BatchNorm1d(128),
            )),
            BasicBlock1D(128, 128, 1)
        )
        self.layer3 = torch.nn.Sequential(
            BasicBlock1D(
                128, 256, 2, 
                downsample=torch.nn.Sequential(
                    conv1x1(128, 256, stride = 2),
                    torch.nn.BatchNorm1d(256),
            )),
            BasicBlock1D(256, 256, 1)
        )
        self.layer4 = torch.nn.Sequential(
            BasicBlock1D(
                256, 512, 2, 
                downsample=torch.nn.Sequential(
                    conv1x1(256, 512, stride = 2),
                    torch.nn.BatchNorm1d(512),
            )),
            BasicBlock1D(512, 512, 1)
        )
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(512, out_features)

    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


def test_resnet1d():
    in_ten = csv_to_tensor("./sample_data/1007233480.csv")
    in_ten = in_ten.unsqueeze(0)
    net = ResNet1D()

    out = net(in_ten)
    print(out)


class LitConvNet1D(LitModel):
    def __init__(self, *args, **kwargs):
        net = ConvNet1D()
        super(LitConvNet1D, self).__init__(net, *args, **kwargs)


if __name__ == "__main__":
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from utilities import summarize_model

    lr = 1e-2
    opt = torch.optim.Adam
    model_name = "LitConvNet1D"
    model = LitConvNet1D(learning_rate=lr, optimizer=opt)
    summarize_model(model)
    print(model.net)

    print(model(torch.rand(2, 10, 600000)))

    full_df = pd.read_csv("./sample_data/train.csv")
    data_dir = "../train/"

    data = Volcano1DDataModule(
        full_df, data_dir, augmenter=None, batch_size=64, maxmem=5e7
    )

    optim_name = opt.__name__
    run_name = f"{model_name}_{optim_name}_{lr}"
    print(f">>>>>>>>>>>>>> {run_name}")
    if not "Stopping Enabled":
        stopper = EarlyStopping(
            monitor="val_loss", verbose=True, patience=20, mode="min"
        )
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

    # trainer.tune(model, data)
    trainer.fit(model, data)
