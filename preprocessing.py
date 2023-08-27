from collections import Counter

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


SAMPLING_RATE = 2  # Sampling Rate: Average 2 samples (rows) per second
WINDOW_SIZE = 14  # ceil(14 * SAMPLING_RATE)  # Best Window Size: 14 seconds


def read_csv(path):
    data = pd.read_csv(path, sep=',', index_col=False)
    data.columns = ['class', 'gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
    return data


def _accelerometer_data(data):
    return data.filter(like='acc', axis=1)


def split_sensors_data(data):
    df = pd.DataFrame(data)
    return _accelerometer_data(df), _gyroscope_data(df)


def _gyroscope_data(data):
    return data.filter(like='gyro', axis=1)


def remove_labels(data):
    return np.delete(data, 0, 1)  # 0: index of label column,  1: delete over the columns


def get_labels(data):
    return data['class']


def sliding_window(data):
    labels = get_labels(data)
    data = remove_labels(data)
    windows_number = len(data) - WINDOW_SIZE + 1

    windowed_data = np.zeros((windows_number, WINDOW_SIZE, data.shape[1]), np.float32)  # data.shape[1] quante sono le colonne + np.float32 data type
    windowed_labels = np.zeros((windows_number, WINDOW_SIZE), np.int8)
    for i in range(windows_number):
        index_range = range(i,i + WINDOW_SIZE)
        windowed_data[i] = data[index_range]

        # Majority rule for selecting the label for the window
        windowed_labels[i] = Counter(labels[index_range]).most_common(1)[0][0]   #return a list of the n most common elements and their counts from the most common to the least.

    return windowed_data, windowed_labels


def data_split(data, label):
    return train_test_split(data, label, train_size=0.8, test_size=0.2)
