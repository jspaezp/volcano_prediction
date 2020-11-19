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

# TODO this name is misleading because it is not really a loader
class tensorLoader(Dataset):
    def __init__(self, train_df, filepath):
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
        for f in self.db:
            # print(f)
            assert f["path"].is_file
        print("Validation Done")

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        item = self.db[index]
        data_tensor = torch.load(item["path"])
        # print(data_tensor.shape)
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


def evaluate(net, testloader, outfile = "file.png"):
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
    r2 = polyfit(x_vals, y_vals, 1)["determination"]

    myfig = plt.figure()
    axes= myfig.add_axes([-0.1,-0.1,1,1])
    axes.scatter(x_vals, y_vals, alpha = 0.5)
    myfig.savefig(outfile)

    print(f"R squared in testing is {r2}")
    return expected, predicted


def main(
    train_file="train.csv",
    data_path=Path("./train-tensors"),
    epochs=2,
    iter=2000,
    lr=0.01,
    device=get_default_device(),
    batch_size=16,
):
    print(f"Device that will be used is {device}")

    net = resnet_10r()
    to_device(net, device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    df = pd.read_csv(train_file)
    df_copy = df.copy()
    train_set = df.sample(frac=0.95, random_state=0)
    test_set = df_copy.drop(train_set.index)

    traindata = tensorLoader(train_set, data_path)
    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=batch_size, shuffle=True, num_workers=5
    )
    trainloader = DeviceDataLoader(trainloader, device=device)

    testdata = tensorLoader(test_set, data_path)
    testloader = torch.utils.data.DataLoader(
        testdata, batch_size=batch_size, shuffle=True, num_workers=5
    )
    testloader = DeviceDataLoader(testloader, device=device)

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

            if i >= iter:
                prog_bar.close()
                break

        prog_bar.close()
        expected, predicted = evaluate(net, testloader, f"epoch_{e}.png")

    print("Finished Training")
    return net


if __name__ == "__main__":

    main()
