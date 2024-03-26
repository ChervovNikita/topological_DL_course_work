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
from scipy.ndimage import distance_transform_edt

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


def diagram(image, device, sublevel=True):
    # get height and square image
    h = int(np.sqrt(image.shape[0]))
    image_sq = image.reshape((h,h))

    # create complex
    cmplx = gd.CubicalComplex(dimensions=(h, h), top_dimensional_cells=(sublevel*image))

    # get pairs of critical simplices
    cmplx.compute_persistence()
    critical_pairs = cmplx.cofaces_of_persistence_pairs()
    
    # get essential critical pixel
    bpx0_essential = critical_pairs[1][0][0] // h, critical_pairs[1][0][0] % h

    # get critical pixels corresponding to critical simplices
    try:
        bpx0 = [critical_pairs[0][0][i][0] for i in range(len(critical_pairs[0][0]))]
        dpx0 = [critical_pairs[0][0][i][1] for i in range(len(critical_pairs[0][0]))]
    except IndexError:
        bpx0 = []
        dpx0 = []
        
    try:
        bpx1 = [critical_pairs[0][1][i][0] for i in range(len(critical_pairs[0][1]))]
        dpx1 = [critical_pairs[0][1][i][1] for i in range(len(critical_pairs[0][1]))]
    except IndexError:
        bpx1 = []
        dpx1 = []
    

    flat_image = image_sq.flatten()
    pd0_essential = torch.tensor([[image_sq[bpx0_essential], torch.max(image)]])

    if (len(bpx0)!=0):
        pdb0 = flat_image[bpx0][:, None]
        pdd0 = flat_image[dpx0][:, None]
        pd0 = torch.Tensor(torch.hstack([pdb0, pdd0]))
        pd0 = torch.vstack([pd0, pd0_essential.to(device)])
    else:
        pd0 = pd0_essential

    if (len(bpx1)!=0):
        pdb1 = flat_image[bpx1][:, None]
        pdd1 = flat_image[dpx1][:, None]
        pd1 = torch.Tensor(torch.hstack([pdb1, pdd1]))
    else:
        pd1 = torch.zeros((1, 2))
    
    return pd0, pd1


def process_by_direction(img, alpha):
    X = (math.cos(alpha) - (np.arange(0, img.shape[0]) - (img.shape[0] / 2 - 0.5)) / (img.shape[0] * math.sqrt(2))) * math.cos(alpha) / 2
    Y = (math.sin(alpha) - (np.arange(0, img.shape[1]) - (img.shape[1] / 2 - 0.5)) / (img.shape[1] * math.sqrt(2))) * math.sin(alpha) / 2
    direction_filter = X.reshape(-1, 1) + Y.reshape(1, -1)
    return np.maximum(direction_filter, img)


def process_image(img, filter_params):
    w = int(np.sqrt(img.flatten().shape[0]))
    imgs = [process_by_direction(img.reshape(w, w), alpha) for alpha in filter_params]
    diagrams = []
    for i, img in enumerate(imgs):
        res = diagram(torch.Tensor(img.flatten()))
        for j in range(len(res)):
            if not res[j].shape[0]:
                diagrams.append(torch.zeros(0, 4))
            else:
                diagrams.append(torch.concatenate([res[j], torch.Tensor([[j, filter_params[i]] for _ in range(res[j].shape[0])])], axis=1))

    diagrams = torch.concatenate(diagrams)
    return diagrams


def process_by_conv(img, conv):
    w = int(np.sqrt(img.flatten().shape[0]))
    img = conv(torch.Tensor(img).reshape(1, w, w)).detach()
    diagrams = []
    for i in range(img.shape[0]):
        res = diagram(img[i].flatten())
        for j in range(len(res)):
            if not res[j].shape[0]:
                diagrams.append(torch.zeros(0, 4))
            else:
                diagrams.append(torch.concatenate([res[j], torch.Tensor([[j, i] for _ in range(res[j].shape[0])])], axis=1))
    diagrams = torch.concatenate(diagrams)
    return diagrams


def process_baseline(img):
    diagrams = []
    res = diagram(torch.Tensor(img.flatten()))
    for j in range(len(res)):
        if not res[j].shape[0]:
            diagrams.append(torch.zeros(0, 4))
        else:
            diagrams.append(torch.concatenate([res[j], torch.Tensor([[j, 1] for _ in range(res[j].shape[0])])], axis=1))
    diagrams = torch.concatenate(diagrams)
    return diagrams


def process_cedt(img):
    edt = distance_transform_edt(img > 0.5)
    cedt = edt * (img > 0.5) - edt * (img <= 0.5)
    return process_baseline(cedt)
