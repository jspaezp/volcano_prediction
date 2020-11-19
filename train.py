import torch
import math
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class tensorLoader(Dataset):
    def __init__(self, train_df, filepath):
        my_iter = zip(train_df["segment_id"], train_df["time_to_eruption"])
        db_map = {}
        db = []

        for i, (x, y) in enumerate(my_iter):
            db_map.update({x: i})
            x = str(x)
            y = torch.tensor(math.log10(float(y)))
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
        return data_tensor[0,:,:,:], item["value"]


if __name__ == "__main__":
    import torch.optim as optim
    import torch.nn as nn
    from v_models import resnet_10r

    net = resnet_10r()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_df = pd.read_csv("train.csv")
    # .glob("*.pt")
    traindata = tensorLoader(train_df, Path("./train-tensors"))
    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=4, shuffle=True, num_workers=2
    )

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        tot_running_loss = 0.0
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

            # print statistics
            running_loss += loss.item()
            tot_running_loss += loss.item()

            curr_runn_loss = running_loss / (i + 1)

            prog_bar.set_postfix({'running_loss': curr_runn_loss})

            if i % 200 == 199:  # print every 200 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print("Finished Training")
