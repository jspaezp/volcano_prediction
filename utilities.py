
import random
import time

import numpy as np
import torch


def shuffle_channels(array: torch.Tensor, axis=2) -> torch.Tensor:
    """Shuffle a torch tensor over a dimension

    :param array: torch.Tensor
    :param axis: Dimension over which the array will be shuffled, defaults to 2
    """
    list_array = list(torch.split(array, 1, axis))
    # print([x.shape for x in ret_a])
    random.shuffle(list_array)
    ret_a = torch.cat(list_array, axis)
    # print(ret_a.shape)
    return ret_a


def test_shuffle_channels():
    """Tests that channel shuffling actually works"""
    sample_arr = torch.stack([torch.ones((10, 10)), torch.zeros(10, 10)])
    orig_shape = sample_arr.shape
    for i in [0, 1, 2]:
        assert orig_shape == shuffle_channels(sample_arr, i).shape

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
        shuffle_channels(sample_arr, 0)
    tot_time = time.time() - start_time
    print(f"{tot_time} for 1000 shuffles")


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


# Polynomial Regression
# https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
def polyfit(x, y, degree):
    """Fits a polynomial regression and returns the coefficients"""
    results = {}

    coefficients = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results["polynomial"] = coefficients.tolist()

    # r-squared
    poly_fit = np.poly1d(coefficients)
    # fit values, and mean
    y_hat = poly_fit(x)  # or [p(z) for z in x]
    y_bar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ss_reg = np.sum((y_hat - y_bar) ** 2)
    ss_tot = np.sum((y - y_bar) ** 2)
    results["determination"] = ss_reg / ss_tot

    return results


def test_polyfit():
    polyfit(np.array([1,2,3,4,5]), np.array([2,4,6,8,10]), 1)
    # {'polynomial': [1.9999999999999996, 4.690263559940989e-16], 'determination': 0.9999999999999997}