
import random

from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from utilities import get_default_device, shuffle_channels, to_device


# TODO this name is misleading because it is not really a loader
class tensorLoader(Dataset):
    def __init__(self, train_df, filepath, shuffle_channels=False, device="cpu"):
        self.shuffle_channels = shuffle_channels
        self.device = torch.device(device)

        my_iter = zip(train_df["segment_id"], train_df["time_to_eruption"])
        db_map = {}
        db = []

        for i, (x, y) in enumerate(my_iter):
            db_map.update({x: i})
            x = str(x)

            y = [float(y) / 1e8]

            # TODO consider if adding tensors here is too much GPU memory
            y = torch.tensor(y, device=device)
            db.append({"path": (Path(filepath) / f"{x}.pt"), "value": y})

        self.db = db
        self.db_map = db_map

        print("Validating Paths")
        counter = 0
        for f in self.db:
            # print(f)
            assert f["path"].is_file()
            counter += 1
        print(f"Validation Done for {counter} Files")

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        item = self.db[index]
        file_path = str(item["path"])

        data_tensor = torch.load(file_path, map_location=self.device)[0, :, :, :]

        if self.shuffle_channels:
            if random.uniform(0, 1) < self.shuffle_channels:
                data_tensor = shuffle_channels(data_tensor, 0)

        return data_tensor, item["value"]


# I got this from a blogpost and I lost the tab ... I AM SORRY
class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dataloader, device=get_default_device()):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for batch in self.dataloader:
            yield to_device(batch, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)


def get_dataloaders(
    train_csv_file: str, batch_size: int, data_path: str, device, num_workers: int = 5
):
    """Generates train test and validation datasets using a csv as a template"""

    df = pd.read_csv(train_csv_file)

    train_set, validate_set, test_set = np.split(
        df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
    )

    traindata = tensorLoader(
        pd.concat([train_set, test_set]), data_path, shuffle_channels=0.2, device=device
    )
    testdata = tensorLoader(test_set, data_path, device=device)
    valdata = tensorLoader(validate_set, data_path, device=device)

    trainloader = torch.utils.data.DataLoader(
        traindata,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    testloader = torch.utils.data.DataLoader(
        testdata,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valloader = torch.utils.data.DataLoader(
        valdata,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # TODO check if the device data loader is required

    trainloader = DeviceDataLoader(trainloader, device=device)
    testloader = DeviceDataLoader(testloader, device=device)
    valloader = DeviceDataLoader(valloader, device=device)

    return trainloader, testloader, valloader
