from augmentation import get_default_augmenter
from lightning import get_default_trainer, VolcanoDataLoader, lit10c_Resnet18


def train():
    scripted_transforms = get_default_augmenter()
    trainer = get_default_trainer()
    model = lit10c_Resnet18()

    train_df = ""
    data_dir = ""

    data = VolcanoDataLoader(train_df, data_dir, augmenter=scripted_transforms)
    trainer.tune(model, data)
