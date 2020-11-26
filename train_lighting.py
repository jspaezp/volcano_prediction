import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from augmentation import get_default_augmenter
from lightning import Lit10cResnet50
from dataloaders import VolcanoDataLoader

# DEFAULTS used by the Trainer
checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=4,
    verbose=True,
    monitor="checkpoint_on",
    mode="min",
    prefix="",
)

scripted_transforms = get_default_augmenter()

logger = TensorBoardLogger("tb_logs", name="my_model")
model = Lit10cResnet50()
trainer = pl.Trainer(
    logger=logger,
    callbacks=[EarlyStopping(monitor="val_loss"), checkpoint_callback],
    auto_lr_find=True,
    gpus=0,
)

train_df = ""
data_dir = ""

data = VolcanoDataLoader(train_df, data_dir, augmenter=scripted_transforms)
trainer.tune(model, data)
