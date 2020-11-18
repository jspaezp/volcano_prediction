import numpy as np
from v_models import resnet_10c
from v_cwt import file_to_cwt_array
import torchvision.models as models
import torch


model = resnet_10c(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=4)

model.eval()

npy_arr = file_to_cwt_array("./sample_data/1000015382.csv")

# Makes the channels be in position 0 instead of position 2
npy_arr2 = np.stack([np.rollaxis(npy_arr, 2, 0)])
tensor_img = torch.from_numpy(npy_arr2)
torch.save(tensor_img, "./sample_data/1000015382.pt")
tensor_img.shape

with torch.no_grad():
    output = model(tensor_img)

print(output[0])
