from augmentation import get_default_augmenter
from lightning import Lit10cResnet50, get_default_trainer, VolcanoDataLoader


def train():
    scripted_transforms = get_default_augmenter()
    trainer = get_default_trainer()
    model = Lit10cResnet50()

    train_df = ""
    data_dir = ""

    data = VolcanoDataLoader(train_df, data_dir, augmenter=scripted_transforms)
    trainer.tune(model, data)
