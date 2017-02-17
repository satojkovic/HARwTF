#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

WISDM_DIR = 'WISDM_ar_v1.1'
RAW_FILE = 'WISDM_ar_v1.1_raw.txt'


def read_data(raw_fpath):
    column_names = [
        'user', 'activity', 'timestamp', 'x-acceleration', 'y-accel', 'z-accel'
    ]
    data = pd.read_csv(raw_fpath, header=None, names=column_names)
    print('Load data:', raw_fpath)
    return data


def feature_normalize(attribute):
    mu = np.mean(attribute, axis=0)
    sigma = np.std(attribute, axis=0)
    result = (attribute - mu) / sigma
    print('Normalized:', result.name)
    return result


def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-acceleration'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-accel'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-accel'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


def main():
    dataset = read_data(os.path.join(WISDM_DIR, RAW_FILE))
    dataset['x-acceleration'] = feature_normalize(dataset['x-acceleration'])
    dataset['y-accel'] = feature_normalize(dataset['y-accel'])
    dataset['z-accel'] = feature_normalize(dataset['z-accel'])

    for activity in np.unique(dataset['activity']):
        subset = dataset[dataset['activity'] == activity][:180]
        plot_activity(activity, subset)


if __name__ == '__main__':
    main()
