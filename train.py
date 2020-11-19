import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd


class tensorLoader(Dataset):
    def __init__(self, train_df, filepath):
        my_iter = zip(train_df["segment_id"], train_df["time_to_eruption"])
        db = {}

        for x, y in my_iter:
            # print(f"x = {x}, y = {y}, filepath = {filepath}")
            x = str(x)
            y = str(y)
            db.update({x: {"path": (Path(filepath) / f"{x}.pt"), "value": y}})
        
        self.db = db

        print("Validating Paths")
        for f in self.db.values():
            # print(f)
            assert f["path"].is_file
        print("Validation Done")

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        item = list(self.db.items())[index][1]
        print(item)
        return torch.load(item["path"]), torch.tensor(float(int(item["value"])))


if __name__ == "__main__":
    import torch.optim as optim
    import torch.nn as nn
    from v_models import resnet_10r

    net = resnet_10r()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_df = pd.read_csv("train.csv")
    # .glob("*.pt")
    trainloader = tensorLoader(train_df, Path("./train-tensors"))

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print("Finished Training")
