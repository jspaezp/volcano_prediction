import torch
import math
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
from v_models import resnet_10r
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# TODO this name is misleading because it is not really a loader
class tensorLoader(Dataset):
    def __init__(self, train_df, filepath, shuffle_channels=False):
        self.shuffle_channels = shuffle_channels
        self.shuffler = nn.ChannelShuffle(2)

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

        print("Validating Paths\n")
        counter = 0
        for f in self.db:
            # print(f)
            assert f["path"].is_file
            counter += 1
        print(f"Validation Done for {counter} Files\n")

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        item = self.db[index]
        data_tensor = torch.load(item["path"])
        # print(data_tensor.shape)
        if self.shuffle_channels == True:
            data_tensor = self.shuffler(data_tensor)

        return data_tensor[0, :, :, :], item["value"]


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


def evaluate(net, testloader, outfile="file.png", prefix="Train", epoch=0, writer=None):
    plt.ioff()

    print("Started Evaluation")
    expected = []
    predicted = []
    with torch.no_grad():
        prog_bar = tqdm(testloader)
        for data in prog_bar:
            images, labels = data
            outputs = net(images)
            predicted.append(outputs)
            expected.append(labels)
        prog_bar.close()

    x_vals = np.concatenate([x.cpu().numpy() for x in expected], axis=None).flatten()
    y_vals = np.concatenate([x.cpu().numpy() for x in predicted], axis=None).flatten()

    mae = np.mean(np.abs(x_vals - y_vals))
    r2 = polyfit(x_vals, y_vals, 1)["determination"]

    metrics = {"mae": mae, "r2": r2}

    if writer is not None:
        writer.add_scalar(f"R2/{prefix}", r2, epoch)
        writer.add_scalar(f"MAE/{prefix}", mae, epoch)

    myfig = plt.figure()
    axes = myfig.add_axes([0.1, 0.1, 1, 1])
    axes.scatter(x_vals, y_vals, alpha=0.5)
    myfig.savefig(outfile)

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
    # Train Loop
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        epoch_loss = 0.0
        curr_running_loss = 0.0
        prog_bar = tqdm(trainloader)

        for i, data in enumerate(prog_bar, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # print(inputs.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            flat_out = outputs.cpu().detach().numpy().flatten()[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()

            curr_epoch_loss = epoch_loss / (i + 1)

            prog_bar.set_postfix(
                {
                    "e": epoch,
                    "epoch_loss": curr_epoch_loss,
                    "running_loss": curr_running_loss,
                    "last_out": flat_out,
                }
            )

            if i % 100 == 99:  # print every 200 mini-batches
                curr_running_loss = running_loss / 100
                # print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, curr_running_loss))
                running_loss = 0.0

            if i >= niter:
                prog_bar.close()
                break

        prog_bar.close()

        # Model evaluations
        expected, predicted, test_metrics = evaluate(
            net,
            testloader,
            f"{prefix}_epoch_{epoch}.png",
            prefix=f"{prefix}_test",
            epoch=epoch,
            writer=writer,
        )
        expected, predicted, val_metrics = evaluate(
            net,
            valloader,
            f"{prefix}_val_epoch_{epoch}.png",
            prefix=f"{prefix}_val",
            epoch=epoch,
            writer=writer,
        )

        # Save Model
        checkpoint_path = Path(out_dir)
        checkpoint_path = (
            checkpoint_path / f"{prefix}_l_{epoch_loss / (i + 1):.4f}_e_{epoch}.pt"
        )
        torch.save(net.state_dict(), checkpoint_path)

    print("Finished Training")
    return net


def train(
    net,
    train_file="train.csv",
    data_path=Path("./train-tensors"),
    epochs=2,
    niter=2000,
    lr=0.01,
    device=get_default_device(),
    batch_size=16,
    checkpoint_dir=".",
    prefix="resnet16",
):
    print(f"Device that will be used is {device}")

    # Tensorboard settings
    writer = SummaryWriter()

    # Network settings
    to_device(net, device)

    # Optimizer and loss settings
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Dataset Loader Settings
    df = pd.read_csv(train_file)
    df_copy = df.copy()
    train_set, validate_set, test_set = np.split(
        df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
    )

    traindata = tensorLoader(
        pd.concat([train_set, test_set]), data_path, shuffle_channels=True
    )
    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=batch_size, shuffle=True, num_workers=5
    )
    trainloader = DeviceDataLoader(trainloader, device=device)

    testdata = tensorLoader(test_set, data_path)
    testloader = torch.utils.data.DataLoader(
        testdata, batch_size=batch_size, shuffle=True, num_workers=5
    )
    testloader = DeviceDataLoader(testloader, device=device)

    valdata = tensorLoader(validate_set, data_path)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=batch_size, shuffle=True, num_workers=5
    )
    valloader = DeviceDataLoader(valloader, device=device)

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

    train()
