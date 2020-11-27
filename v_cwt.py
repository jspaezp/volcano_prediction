import pandas as pd
import numpy as np

import pywt

import torch
from torch.nn.functional import interpolate


def reshape_coef_arr(coefs, shape=(224, 224), prepacking=1000):
    n = coefs.shape[1] // prepacking

    # split requires a precise division, so this removes any leftovers
    coefs = coefs[:, 0 : n * prepacking]
    pre_splits = np.split(coefs, prepacking, 1)

    coef2 = np.stack([x.max(axis=1) for x in pre_splits], 1)
    tensor = torch.from_numpy(coef2).unsqueeze(0).unsqueeze(0)
    resized = interpolate(tensor, shape, mode="bilinear", align_corners=False)
    return resized[0, 0, ...].detach().clone()


def default_cwt(sequence):
    # Each sensor seems to be 16 bit, vals from -32k to +32k

    #  Each file contains ten minutes of logs from ten different sensors arrayed around a volcano
    #  So I will make a linear interpolation in seconds (10*60), interval is 0.1s

    # with this wavelet (morl) a sequence from 2 to 128 covers frequencies from 40 to 0.6hz
    scales = 2 ** np.linspace(1, 9, 50)
    data_section = np.nan_to_num(sequence.to_numpy().astype("float32"))
    data_section = data_section / 2.0 ** 16
    coef, freqs = pywt.cwt(data=data_section, scales=scales, wavelet="morl")
    return coef, freqs


def pd_to_cwt_array(data_sensors: pd.DataFrame, reshape_size=(224, 224)):

    trans = []
    for col in list(data_sensors):
        coef, freqs = default_cwt(data_sensors[col])
        trans.append(np.abs(coef))

    tens = torch.stack(
        [reshape_coef_arr(x, reshape_size, prepacking=500) for x in trans]
    )

    return tens


def file_to_cwt_array(filename, reshape_size=(256, 256), verbose=True):
    tens = pd_to_cwt_array(pd.read_csv(filename), reshape_size=reshape_size)
    if verbose:
        print(tens.shape)

    return tens


def file_to_cwt_array_disk(
    filename, out_filename, reshape_size, verbose=True, dry=False
):
    tens = file_to_cwt_array(filename, reshape_size, verbose)
    tens = tens.unsqueeze(0)

    if verbose:
        message = f"Saving tensor of shape {tens.shape} to {out_filename}"
        if dry:
            message = "Dry Run, Not " + message

        print(message)
    torch.save(tens, out_filename)


if __name__ == "__main__":
    import time

    st = time.time()
    for _ in range(10):
        file_to_cwt_array("sample_data/1007233480.csv", (512, 512))
    print(f"Takes in average {(time.time() - st)/10} seconds per image")
