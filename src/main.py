import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os import listdir as ld
import random


def ReLu(x):
    if x<0:
        return(0)
    return(x)

data = pd.read_csv('data/mnist_train.csv')

data = np.array(data)
m, n = data.shape

np.random.shuffle(data)

data_dev = data[0:1000].T

