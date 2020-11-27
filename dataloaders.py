from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from utilities import get_default_device, to_device


class AugmentedDataset(Dataset):
    def __init__(self, dataset, augmenter):
        self.augmenter = augmenter
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset.__getitem__(index)
        if self.augmenter is not None:
            x = self.augmenter(x.detach().clone())
        return x, y

    def __len__(self):
        return len(self.dataset)


class TensorDispatcher(Dataset):
    def __init__(self, data_tensor, response_tensor, augmenter=None):
        self.augmenter = augmenter
        assert data_tensor.shape[0] == response_tensor.shape[0]

        self.data_tensor = data_tensor
        self.response_tensor = response_tensor

    def __len__(self):
        return self.data_tensor.shape[0]

    def __getitem__(self, index):
        x = self.data_tensor[index, ...].detach().clone()
        y = self.response_tensor[index, ...].detach().clone()

        if self.augmenter is not None:
            x = self.augmenter(image=x)

        return x.float(), y


def test_TensorDispatcher():
    x = torch.ones((10, 50, 50))
    y = torch.zeros((10, 1))
    i = 0

    TD = TensorDispatcher(x, y)
    for i, (x, y) in enumerate(TD):
        continue

    assert torch.all(x == torch.ones(50, 50))
    assert torch.all(y == torch.zeros(1))
    assert i == 9


# TODO this name is misleading because it is not really a loader
class GreedyTensorLoader(TensorDispatcher):
    def __init__(self, train_df, data_dir, maxmem=5e9, augmenter=None):
        print("Setting up greedy tensor loader")
        my_iter = zip(train_df["segment_id"], train_df["time_to_eruption"])
        spectra = []
        responses = []

        mem = 0
        for i, (x, y) in enumerate(my_iter):
            filepath = Path(data_dir) / f"{x}.pt"
            data_tensor = torch.load(str(filepath))[0, :, :, :].half()
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

        super(GreedyTensorLoader, self).__init__(
            data_tensor=torch.stack(spectra),
            response_tensor=torch.stack(responses),
            augmenter=augmenter,
        )

    def __len__(self):
        return super(GreedyTensorLoader, self).__len__()

    def __getitem__(self, index):
        return super(GreedyTensorLoader, self).__getitem__(index)


def test_GreedyTensorLoader():
    tiny_df = pd.DataFrame({"segment_id": [1000015382] * 2, "time_to_eruption": [1, 1]})
    DL = GreedyTensorLoader(train_df=tiny_df, data_dir="sample_data")
    i = 0

    for i, (x, y) in enumerate(DL):
        print(x.shape)
        print(y.shape)

    assert i == 1


class tensorLoader(Dataset):
    def __init__(self, train_df, filepath, augmenter=None):
        self.augmenter = augmenter

        my_iter = zip(train_df["segment_id"], train_df["time_to_eruption"])
        db_map = {}
        db = []

        for i, (x, y) in enumerate(my_iter):
            db_map.update({x: i})
            x = str(x)

            y = [float(y) / 1e8]

            # TODO consider if adding tensors here is too much GPU memory
            y = torch.tensor(y)
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

        data_tensor = torch.load(file_path)[0, :, :, :]

        if self.augmenter is not None:
            data_tensor = torch.from_numpy(self.augmenter(image=data_tensor))

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
    train_csv_file: str,
    batch_size: int,
    data_path: str,
    device,
    num_workers: int = 5,
    augmenter=None,
):
    """Generates train test and validation datasets using a csv as a template"""

    df = pd.read_csv(train_csv_file)

    train_set, validate_set, test_set = np.split(
        df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
    )

    traindata = tensorLoader(
        pd.concat([train_set, test_set]), data_path, augmenter=augmenter
    )
    testdata = tensorLoader(test_set, data_path)
    valdata = tensorLoader(validate_set, data_path)

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


if __name__ == "__main__":
    test_TensorDispatcher()
    test_GreedyTensorLoader()
