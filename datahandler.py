import numpy as np


def data_norm(data):
    min_old = min(data)
    max_old = max(data)
    min_new = 0.1
    max_new = 0.9
    data2 = np.array(data)
    return (data2 - min_old) / (max_old - min_old) * (max_new - min_new)


def generate_series(data, series):
    x = []
    y = []
    for i in range(len(data)-series):
        x.append(data[i:i + series])
        y.append(data[i + series])
    return x, y
