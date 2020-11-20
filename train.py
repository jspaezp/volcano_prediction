import time
from pathlib import Path

from tqdm.auto import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from dataloaders import get_dataloaders
from utilities import polyfit, to_device, get_default_device

from typing import Tuple


def evaluate(
    net, testloader, outfile="file.png", prefix="Train", verbose=True
) -> Tuple[np.array, np.array, dict]:
    """Evaluate a model

    :param net: Model to test
    :param testloader: Dataloader that provides the data
    :param outfile: Image file where results will be displayed, defaults to "file.png"
    :param prefix: Text to be added to the results, defaults to "Train"
    :param verbose: Wether to display more output, defaults to True
    :return: Returns 3 elements, the expected, the predicted and a dictionary with metrics
    """
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
    rsq = polyfit(x_vals, y_vals, 1)["determination"]

    metrics = {"mae": mae, "r2": rsq, "mse": mse}

    plt.ioff()
    my_figure = plt.figure()
    axes = my_figure.add_axes([0.1, 0.1, 1, 1])
    axes.scatter(x_vals, y_vals, alpha=0.5)
    my_figure.savefig(outfile)
    plt.close(my_figure)

    print(
        f">>{prefix} Evaluation Results: Rsq: {rsq:.4f}, MAE: {mae:.5f}, MSE: {mse:.5f}"
    )
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
    print(f"\n>>>> Prefix: {prefix} <<<<\n")
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
            flat_out = outputs.data[0]

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()

            curr_epoch_loss = epoch_loss / (i + 1)

            prog_bar.set_postfix(
                {"e": epoch, "epoch_loss": curr_epoch_loss, "last_out": flat_out,}
            )

            if i >= niter:
                prog_bar.close()
                break

        prog_bar.close()
        epoch_end_time = time.time()
        walltime_epoch_end = epoch_end_time - start_time

        print("Evaluating")

        # Model evaluations
        _, _, test_metrics = evaluate(
            net,
            testloader,
            f"{prefix}_epoch_{epoch}.png",
            prefix=f"{prefix}_test",
            verbose=False,
        )
        writer.add_scalar(
            "R2/Test", test_metrics["r2"], epoch, walltime=walltime_epoch_end
        )
        writer.add_scalar(
            "MAE/Test", test_metrics["mae"], epoch, walltime=walltime_epoch_end
        )
        writer.add_scalar(
            "MSE/Test", test_metrics["mse"], epoch, walltime=walltime_epoch_end
        )

        _, _, val_metrics = evaluate(
            net,
            valloader,
            f"{prefix}_val_epoch_{epoch}.png",
            prefix=f"{prefix}_val",
            verbose=False,
        )
        writer.add_scalar(
            "R2/Val", val_metrics["r2"], epoch, walltime=walltime_epoch_end
        )
        writer.add_scalar(
            "MAE/Val", val_metrics["mae"], epoch, walltime=walltime_epoch_end
        )
        writer.add_scalar(
            "MSE/Val", val_metrics["mse"], epoch, walltime=walltime_epoch_end
        )

        # Save Model
        checkpoint_path = Path(out_dir)
        checkpoint_path = (
            checkpoint_path / f"L_{val_metrics['mse']:.4f}_{prefix}_e_{epoch}.pt"
        )
        torch.save(net.state_dict(), checkpoint_path)

    minutes_taken = (time.time() - start_time) / 60
    print(f">>> Finished Training, took {minutes_taken:.3f} minutes to run <<<\n")
    return net


def train(
    net,
    prefix="network",
    train_file="train.csv",
    data_path=Path("./train-tensors"),
    epochs=2,
    niter=2000,
    lr=0.01,
    device=None,
    batch_size=16,
    checkpoint_dir=".",
):

    if device is None:
        device = get_default_device()

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
