import torch
import cv2
import matplotlib.pyplot as plt


def show_tensor(a):
    assert len(a.shape) == 3
    # Tensor of shape (channels, w, h)
    ret_a = list(torch.split(a, 1, 0))
    ret_a = [x.cpu().numpy() for x in ret_a]
    full = cv2.vconcat(ret_a)
    plt.imshow(-full)
    plt.show()
