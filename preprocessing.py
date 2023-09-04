from collections import Counter

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


SAMPLING_RATE = 2  # Sampling Rate: Average 2 samples (rows) per second
WINDOW_SIZE = 14  # ceil(14 * SAMPLING_RATE)  # Best Window Size: 14 seconds


def prepare_dataset():
    normal_driving_dataset = pd.read_csv("./dataset/train_motion_data.csv", sep=',', index_col=False)
    normal_driving_dataset.loc[normal_driving_dataset['Class'] != "AGGRESSIVE"]
    df = pd.DataFrame(columns=['class', 'gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ'])
    df['class'] = [5] * len(normal_driving_dataset)     # class label: normal driving
    df['gyroX'] = normal_driving_dataset['GyroX']
    df['gyroY'] = normal_driving_dataset['GyroY']
    df['gyroZ'] = normal_driving_dataset['GyroZ']
    df['accX'] = normal_driving_dataset['AccX']
    df['accY'] = normal_driving_dataset['AccY']
    df['accZ'] = normal_driving_dataset['AccZ']

    dataset = read_csv("./dataset/sensor_raw.csv")
    dataset['gyroX'] = dataset['gyroX'].round(7)
    dataset['gyroY'] = dataset['gyroY'].round(7)
    dataset['gyroZ'] = dataset['gyroZ'].round(7)
    dataset['accX'] = dataset['accX'].round(7)
    dataset['accY'] = dataset['accY'].round(7)
    dataset['accZ'] = dataset['accZ'].round(7)
    dataset.to_csv("./dataset/final_dataset.csv", index=False, mode="w")
    df.to_csv("./dataset/final_dataset.csv", index=False, mode="a", header=False)       #mode = a (append); header = False no row with columns' names

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


def get_data(data):
    return np.delete(data, 0, 1)  # 0: index of label column,  1: delete over the columns


def get_labels(data):
    return data['class']

# The dataset's labels belongs to [1, 5] range, but I need them to belong to [0, 4]
def scale_labels(labels):
    return labels-1


def sliding_window(data):
    labels = scale_labels(get_labels(data))
    data = get_data(data)
    windows_number = len(data) - WINDOW_SIZE + 1

    windowed_data = np.zeros((windows_number, WINDOW_SIZE, data.shape[1]), np.float32)  # data.shape[1] quante sono le colonne + np.float32 data type
    windowed_labels = np.zeros(windows_number, np.int8)
    for i in range(windows_number):
        index_range = range(i,i + WINDOW_SIZE)
        windowed_data[i] = data[index_range]

        # Majority rule for selecting the label for the window
        windowed_labels[i] = Counter(labels[index_range]).most_common(1)[0][0]   #return a list of the n most common elements and their counts from the most common to the least.

    return windowed_data, windowed_labels


def data_split(data, label):
    return train_test_split(data, label, train_size=0.8, test_size=0.2)


# to create the final and complete dataset with 5 labels
# prepare_dataset()