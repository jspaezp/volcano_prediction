import pandas as pd
import numpy as np

import pywt
import cv2


def pd_to_cwt_list(data_sensors: pd.DataFrame):
    # Each sensor seems to be 16 bit, vals from -32k to +32k

    #  Each file contains ten minutes of logs from ten different sensors arrayed around a volcano
    #  So I will make a linear interpolation in seconds (10*60), interval is 0.1s

    # with this wavelet (morl) a sequence from 2 to 128 covers frequencies from 40 to 0.6hz
    sequence = np.linspace(2, 128, 50)

    trans = []

    for col in list(data_sensors):
        print(col)
        data_section = np.nan_to_num(data_sensors[col].to_numpy().astype("float32"))
        coef, freqs = pywt.cwt(data=data_section / 2.**16, scales=sequence, wavelet="morl")

        trans.append(np.abs(coef))

    return trans


def file_to_cwt_list(filename):
    data_sensors = pd.read_csv(str(filename))
    return pd_to_cwt_list(data_sensors)


def file_to_cwt_array(filename, reshape_size=(256, 256)):
    cwt_list = file_to_cwt_list(filename)
    arrays = [
        cv2.resize(x, reshape_size=reshape_size).astype(np.float32) for x in cwt_list
    ]
    ret = np.stack(arrays, axis=-1)
    print(ret.shape)

    return ret
