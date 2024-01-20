import torch
import torchvision
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from ripser import lower_star_img
from ripser import Rips
import persim
import diagram2vec
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance as dist_w
from IPython.display import clear_output
import os


def get_data(path):
    mnist = loadmat("path")
    X = mnist["data"].T
    y = mnist["label"][0]
    return X, y


if __name__ == "__main__":
    # path = os.
    pass

