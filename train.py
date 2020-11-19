import torch
import math
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
from v_models import resnet_10r

class tensorLoader(Dataset):
    def __init__(self, train_df, filepath):
        my_iter = zip(train_df["segment_id"], train_df["time_to_eruption"])
        db_map = {}
        db = []

        for i, (x, y) in enumerate(my_iter):
            db_map.update({x: i})
            x = str(x)
            y = [math.log10(float(y))]
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
        return data_tensor[0,:,:,:], item["value"]


def evaluate(net, testloader):
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

    return expected, predicted
            

def main(train_file = "train.csv", data_path = Path("./train-tensors"), epochs = 2, iter = 2000):
    net = resnet_10r()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    df = pd.read_csv(train_file)
    df_copy = df.copy()
    train_set = df.sample(frac=0.95, random_state=0)
    test_set = df_copy.drop(train_set.index)

    # .glob("*.pt")
    traindata = tensorLoader(train_set, data_path)
    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=1, shuffle=True, num_workers=5
    )

    testdata = tensorLoader(test_set, data_path)
    testloader = torch.utils.data.DataLoader(
        testdata, batch_size=1, shuffle=True, num_workers=5
    )

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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()

            curr_epoch_loss = epoch_loss / (i + 1)

            prog_bar.set_postfix({'epoch_loss': curr_epoch_loss, 'running_loss': curr_running_loss})

            if i % 100 == 99:  # print every 200 mini-batches
                curr_running_loss = running_loss / 100
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, curr_running_loss))
                running_loss = 0.0

            if i >= iter:
                break

        expected, predicted = evaluate(net, testloader)

    print("Finished Training")
    return net

if __name__ == "__main__":

    main()

