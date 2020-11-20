import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import time
import torch.optim as optim
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import random


def shuffle_channels(a, axis=2):
    ret_a = list(torch.split(a, 1, axis))
    # print([x.shape for x in ret_a])
    random.shuffle(ret_a)
    ret_a = torch.cat(ret_a, axis)
    # print(ret_a.shape)
    return ret_a


def test_shuffle_channels():
    samp_arr = torch.stack([torch.ones((10, 10)), torch.zeros(10, 10)])
    orig_shape = samp_arr.shape
    for i in [0, 1, 2]:
        assert orig_shape == shuffle_channels(samp_arr, i).shape

    # torch.Size([4, 10, 10])
    img_like_array = torch.stack([x * torch.ones((10, 10)) for x in range(4)])
    shuffled_img_like = shuffle_channels(img_like_array, 0)
    for i in range(4):
        assert torch.std(shuffled_img_like[i, :, :]) == 0

    img_like_array = torch.stack(
        [x * torch.arange(100).reshape((10, 10)) for x in range(4)]
    )
    shuffled_img_like = shuffle_channels(img_like_array, 0)
    # shuffled_img_like
    # img_like_array
    start_time = time.time()
    for i in range(1000):
        shuffle_channels(samp_arr, 0)
    tot_time = time.time() - start_time
    print(f"{tot_time} for 1000 shuffles")


# TODO this name is misleading because it is not really a loader
class tensorLoader(Dataset):
    cache = {}

    def __init__(self, train_df, filepath, shuffle_channels=False, cache=False):
        self.shuffle_channels = shuffle_channels
        self.cache = cache

        my_iter = zip(train_df["segment_id"], train_df["time_to_eruption"])
        db_map = {}
        db = []

        for i, (x, y) in enumerate(my_iter):
            db_map.update({x: i})
            x = str(x)
            # TODO: check if dividing by the mean order of magnitude would be
            # better than log scaling the out...
            # y = [math.log10(float(y))]

            y = [float(y) / 1e8]
            y = torch.tensor(y)
            db.append({"path": (Path(filepath) / f"{x}.pt"), "value": y})

        self.db = db
        self.db_map = db_map

        print("Validating Paths")
        counter = 0
        for f in self.db:
            # print(f)
            assert f["path"].is_file
            counter += 1
        print(f"Validation Done for {counter} Files")

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        item = self.db[index]
        file_path = item["path"]

        if self.cache and file_path not in tensorLoader.cache:
            # Tensors are of shape
            # (1, 10, x, x), being 10 channels
            tensorLoader.cache[file_path] = torch.load(file_path)[0, :, :, :]

        data_tensor = tensorLoader.cache[file_path]

        if self.shuffle_channels:
            if random.uniform(0,1) < self.shuffle_channels:
                data_tensor = shuffle_channels(data_tensor, 0)

        return data_tensor, item["value"]

    @classmethod
    def flush_cache(cls):
        tensorLoader.cache = {}
        print("Flushed Cache")


# GPU usage gotten from:
# https://medium.com/analytics-vidhya/training-deep-neural-networks-on-a-gpu-with-pytorch-2851ccfb6066
# torch.cuda.is_available()
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device=get_default_device()):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# Polynomial Regression
# https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results["polynomial"] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results["determination"] = ssreg / sstot

    return results


def evaluate(net, testloader, outfile="file.png", prefix="Train", verbose=True):
    expected = []
    predicted = []
    with torch.no_grad():
        if verbose:
            data_iterator = tqdm(testloader)
        else:
            data_iterator = testloader

        for data in data_iterator:
            images, labels = data
            outputs = net(images)
            predicted.append(outputs)
            expected.append(labels)

        if verbose:
            data_iterator.close()

    x_vals = np.concatenate([x.cpu().numpy() for x in expected], axis=None).flatten()
    y_vals = np.concatenate([x.cpu().numpy() for x in predicted], axis=None).flatten()

    mse = np.mean((x_vals - y_vals) ** 2)
    mae = np.mean(np.abs(x_vals - y_vals))
    r2 = polyfit(x_vals, y_vals, 1)["determination"]

    metrics = {"mae": mae, "r2": r2, "mse": mse}

    plt.ioff()
    myfig = plt.figure()
    axes = myfig.add_axes([0.1, 0.1, 1, 1])
    axes.scatter(x_vals, y_vals, alpha=0.5)
    myfig.savefig(outfile)
    plt.close(myfig)

    print(f"\n>>{prefix} Evaluation Results: Rsq: {r2:.4f}, MAE: {mae:.5f}\n")
    return expected, predicted, metrics


def train_loop(
    net,
    trainloader,
    testloader,
    valloader,
    criterion,
    optimizer,
    epochs,
    prefix,
    writer=None,
    out_dir=".",
    niter=2000,
):
    print(f"\n\n>>>> Prefix: {prefix} <<<<\n\n")
    start_time = time.time()
    # Train Loop
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"\n>> {prefix} Epoch: {epoch} <<\n")

        i = 0
        running_loss = 0.0
        epoch_loss = 0.0
        prog_bar = tqdm(trainloader)

        for i, data in enumerate(prog_bar, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # print(inputs.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            flat_out = outputs.cpu().detach().numpy().flatten()[0]

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()

            curr_epoch_loss = epoch_loss / (i + 1)

            prog_bar.set_postfix(
                {
                    "e": epoch,
                    "epoch_loss": curr_epoch_loss,
                    "last_out": flat_out,
                }
            )

            if i >= niter:
                prog_bar.close()
                break

        prog_bar.close()

        print("Evaluating")

        # Model evaluations
        expected, predicted, test_metrics = evaluate(
            net,
            testloader,
            f"{prefix}_epoch_{epoch}.png",
            prefix=f"{prefix}_test",
            verbose=False,
        )
        writer.add_scalar(f"Test_R2/{prefix}", test_metrics["r2"], epoch)
        writer.add_scalar(f"Test_MAE/{prefix}", test_metrics["mae"], epoch)
        writer.add_scalar(f"Test_MSE/{prefix}", test_metrics["mse"], epoch)

        expected, predicted, val_metrics = evaluate(
            net,
            valloader,
            f"{prefix}_val_epoch_{epoch}.png",
            prefix=f"{prefix}_val",
            verbose=False,
        )
        writer.add_scalar(f"Val_R2/{prefix}", val_metrics["r2"], epoch)
        writer.add_scalar(f"Val_MAE/{prefix}", val_metrics["mae"], epoch)
        writer.add_scalar(f"Val_MSE/{prefix}", val_metrics["mse"], epoch)

        # Save Model
        checkpoint_path = Path(out_dir)
        checkpoint_path = (
            checkpoint_path / f"L_{epoch_loss / (i + 1):.4f}_{prefix}_e_{epoch}.pt"
        )
        torch.save(net.state_dict(), checkpoint_path)

    minutes_taken = (time.time() - start_time) / 60
    print(f"Finished Training, took {minutes_taken:.3f} minutes to run")
    return net


def get_dataloaders(train_csv_file, batch_size, data_path, device, num_workers=5, cache=False):
    df = pd.read_csv(train_csv_file)

    train_set, validate_set, test_set = np.split(
        df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
    )

    traindata = tensorLoader(
        pd.concat([train_set, test_set]), data_path, shuffle_channels=0.2, cache=cache
    )
    testdata = tensorLoader(test_set, data_path, cache=cache)
    valdata = tensorLoader(validate_set, data_path, cache=cache)

    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        testdata, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    trainloader = DeviceDataLoader(trainloader, device=device)
    testloader = DeviceDataLoader(testloader, device=device)
    valloader = DeviceDataLoader(valloader, device=device)

    return trainloader, testloader, valloader


def train(
    net,
    prefix="network",
    train_file="train.csv",
    data_path=Path("./train-tensors"),
    epochs=2,
    niter=2000,
    lr=0.01,
    device=get_default_device(),
    batch_size=16,
    checkpoint_dir=".",
):
    print(f"Device that will be used is {device}")

    # Tensorboard settings
    writer = SummaryWriter(comment=prefix)

    # Network settings
    to_device(net, device)

    # Optimizer and loss settings
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Dataset Loader Settings
    trainloader, testloader, valloader = get_dataloaders(
        train_file, batch_size, data_path, device
    )

    net = train_loop(
        net=net,
        trainloader=trainloader,
        testloader=testloader,
        valloader=valloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        prefix=prefix,
        writer=writer,
        out_dir=checkpoint_dir,
        niter=niter,
    )
    return net


if __name__ == "__main__":
    pass
    # train()
