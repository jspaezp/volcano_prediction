import os
from utilities import summarize_model
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from dataloaders import GreedyTensorLoader, TensorDispatcher, AugmentedDataset
from v_models import resnet_10r, densenet_10r

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


class LitModel(pl.LightningModule):
    def __init__(
        self, net, optimizer=torch.optim.Adam, learning_rate=1e-4, loss=torch.nn.MSELoss
    ):
        super(LitModel, self).__init__()
        self.optimizer = optimizer
        self.lr = learning_rate
        self.loss = loss()
        self.net = net

    def forward(self, x):
        x = self.net.forward(x)
        return x

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.net.parameters(), lr=(self.lr or self.learning_rate)
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        val_loss = self.loss(yhat, y)
        self.log("val_loss", val_loss)

        return val_loss


class Lit10CResnet(LitModel):
    def __init__(
        self,
        block,
        layers,
        optimizer=torch.optim.Adam,
        learning_rate=1e-4,
        loss=torch.nn.MSELoss,
        *args,
        **kwargs,
    ):
        resnet = resnet_10r(block, layers, *args, **kwargs)
        super(Lit10CResnet, self).__init__(
            resnet, optimizer=optimizer, learning_rate=learning_rate, loss=loss
        )


def lit10c_Resnet18(*args, **kwargs):
    return Lit10CResnet(BasicBlock, [2, 2, 2, 2])


def lit10c_Resnet34(*args, **kwargs):
    return Lit10CResnet(BasicBlock, [3, 4, 6, 3])


def lit10c_Resnet50(*args, **kwargs):
    return Lit10CResnet(Bottleneck, [3, 4, 6, 3])


def lit10c_Resnet101(*args, **kwargs):
    return Lit10CResnet(Bottleneck, [3, 4, 23, 3])


def lit10c_Resnet152(*args, **kwargs):
    return Lit10CResnet(Bottleneck, [3, 8, 36, 3])


def lit10c_resnext50_32x4d(*args, **kwargs):
    return Lit10CResnet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)


def lit10c_Densenet121(*args, **kwargs):
    return Lit10CDensenet(
        growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64
    )


def lit10c_Densenet161(*args, **kwargs):
    return Lit10CDensenet(
        growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96
    )


def lit10c_Densenet169(*args, **kwargs):
    return Lit10CDensenet(
        growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64
    )


def lit10c_Densenet201(*args, **kwargs):
    return Lit10CDensenet(
        growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64
    )


class Lit10CDensenet(LitModel):
    def __init__(
        self,
        optimizer=torch.optim.Adam,
        learning_rate=1e-4,
        loss=torch.nn.MSELoss,
        *args,
        **kwargs,
    ):
        densenet = densenet_10r(*args, **kwargs)
        super(Lit10CDensenet, self).__init__(
            densenet, optimizer=optimizer, learning_rate=learning_rate, loss=loss
        )


class TensorDataLoader(pl.LightningDataModule):
    def __init__(self, x_tensor, y_tensor, batch_size, train_split, augmenter):
        super(TensorDataLoader, self).__init__()
        self.batch_size = batch_size
        self.train_split = train_split
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.augmenter = augmenter

    def setup(self, stage=None):
        data_full = TensorDispatcher(self.x_tensor, self.y_tensor)
        db_size = len(data_full)
        train_size = int(db_size * self.train_split)
        val_size = db_size - train_size
        train, self.val = random_split(data_full, [train_size, val_size])
        self.train = AugmentedDataset(train, self.augmenter)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)


def test_TensorDataLoader():
    # Test that it works
    x_tensor = torch.stack([x * torch.ones((3, 3)) for x in range(50)])
    y_tensor = torch.stack([x * torch.ones((1)) for x in range(50)])

    def samp_augmenter(image):
        return image * torch.rand(1)

    tdl = TensorDataLoader(
        x_tensor=x_tensor,
        y_tensor=y_tensor,
        augmenter=samp_augmenter,
        batch_size=2,
        train_split=0.5,
    )
    tdl.setup()

    tr_dl = tdl.train_dataloader()
    val_dl = tdl.val_dataloader()

    x, x2, y, y2 = (1, 2, 3, 4)

    for x, y in zip(tr_dl, val_dl):
        continue

    for x2, y2 in zip(tr_dl, val_dl):
        continue

    # Test that it actually augments where it is supposed to
    assert not torch.all(x[0] == x2[0])
    # Test that it actually does not when it is not supposed to
    assert torch.all(y[0] == y2[0])

    # TODO write test to check that shuffling happends when it is supposed to
    # print(x2[1])
    # print(x[1])


# TODO decouple the reading with the dataset,
class VolcanoDataLoader(pl.LightningDataModule):
    def __init__(self, train_df, data_dir, batch_size, train_split=0.9, augmenter=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_split = train_split
        self.data_dir = data_dir
        self.train_df = train_df
        self.augmenter = augmenter

    def setup(self, stage=None):
        volcano_data_full = GreedyTensorLoader(self.train_df, self.data_dir)
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


def get_default_trainer(ngpus=0):
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=4,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )
    logger = TensorBoardLogger("tb_logs", name="my_model")
    stopper = EarlyStopping(monitor="val_loss", verbose=True, patience=50)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            stopper,
            checkpoint_callback,
        ],
        auto_lr_find=True,
        gpus=ngpus,
        precision=16,
        progress_bar_refresh_rate=25,
        profiler="simple",
    )

    return trainer


def test_train():
    import pandas as pd

    tiny_df = pd.DataFrame(
        {"segment_id": [1000015382] * 4, "time_to_eruption": [1] * 4}
    )
    DL = VolcanoDataLoader(
        train_df=tiny_df, data_dir="sample_data", batch_size=2, train_split=0.5
    )
    model = lit10c_Resnet18()
    trainer = pl.Trainer(max_epochs=2, profiler="simple")

    trainer.fit(model, DL)


if __name__ == "__main__":
    test_train()
    test_TensorDataLoader()
    summarize_model(lit10c_Resnet18())
    summarize_model(lit10c_Resnet34())
    summarize_model(lit10c_Resnet50())
    summarize_model(lit10c_Resnet101())
    summarize_model(lit10c_Resnet152())
    summarize_model(lit10c_resnext50_32x4d())
    summarize_model(lit10c_Densenet121())
    summarize_model(lit10c_Densenet161())
    summarize_model(lit10c_Densenet169())
    summarize_model(lit10c_Densenet201())
