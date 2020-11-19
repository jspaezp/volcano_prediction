import pandas as pd
import numpy as np

import pywt
import cv2

import matplotlib.pyplot as plt


def pd_to_cwt_list(data_sensors: pd.DataFrame):
    #  Each file contains ten minutes of logs from ten different sensors arrayed around a volcano
    # So I will make a linear interpolation in seconds (10*60), interval is 0.1s
    # sequence = np.linspace(0, 10*60, len(data_sensors))

    # [0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 64, 7: 128, 8: 256, 9: 512]
    sequence = np.arange(1, 2 ** 8)

    trans = []

    for col in list(data_sensors):
        print(col)
        data_section = np.nan_to_num(data_sensors[col].to_numpy().astype("float32"))
        coef, freqs = pywt.cwt(data=data_section, scales=sequence, wavelet="morl")

        trans.append(coef)

    return trans


def file_to_cwt_list(filename):
    data_sensors = pd.read_csv(str(filename))
    return pd_to_cwt_list(data_sensors)


def norm_arr(x, reshape_size=(256, 256)):
    resized = cv2.resize(x, reshape_size)
    abs_arr = np.abs(np.nan_to_num(resized))

    # I am adding 0.01 to prevent division by 0 issues
    abs_max = abs_arr / (np.std(abs_arr) + 0.01)

    # This would make the limit from 0 to 1
    # And censoring the upper limit at 4 standard deviations
    abs_max = abs_max / 4
    abs_max[abs_max > 1] = 1

    return abs_max


def file_to_cwt_array(filename, reshape_size=(256, 256)):
    cwt_list = file_to_cwt_list(filename)
    arrays = [
        norm_arr(x, reshape_size=reshape_size).astype(np.float32) for x in cwt_list
    ]
    ret = np.stack(arrays, axis=-1)
    print(ret.shape)

    return ret


def visualize_cwt_list(cwt_list):
    full = cv2.vconcat(cwt_list)
    viz_full_max = np.max(np.abs(np.nan_to_num(full))) ** (1 / 3)
    viz_full = 128 + 127 * np.nan_to_num(full) / viz_full_max
    viz_full = viz_full.astype(np.uint8)
    im_color = cv2.applyColorMap(viz_full, cv2.COLORMAP_BONE)
    return im_color


def foo():
    arr_l = file_to_cwt_array("./sample_data/1000015382.csv")
    viz = visualize_cwt_list(arr_l)

    cv2.imwrite("img.png", viz)

    import matplotlib.pyplot as plt

    _ = plt.hist(viz_full, bins="auto")
    plt.show()
