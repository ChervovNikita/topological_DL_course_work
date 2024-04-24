from datasets import *
from transforms import *

from __future__ import print_function

%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from ripser import lower_star_img
from ripser import Rips
vr = Rips()
from gtda.homology import VietorisRipsPersistence

import persim
import diagram2vec

from scipy.ndimage import gaussian_filter

from sklearn.datasets import make_circles
from sklearn.manifold import MDS

from gtda.diagrams import PersistenceEntropy, PersistenceImage, BettiCurve

import pickle
from tqdm import tqdm

import torch
from torch.nn import Linear
from torch.nn.functional import relu

from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import pandas as pd
import os
from PIL import Image

from sklearn.model_selection import cross_val_score


### MNIST


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST('../data/mnist/base_mnist_train_data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST('../data/mnist/base_mnist_test_data', train=False, download=True, transform=transform)
X_train = mnist_train.data.numpy() / 255
X_test = mnist_test.data.numpy() / 255
y_train = mnist_train.targets.numpy()
y_test = mnist_test.targets.numpy()

def push_diagrams_mnist(train_diagrams, test_diagrams, train_labels, test_labels, name):
    train_dataset = DiagramsDataset(train_diagrams, train_labels)
    test_dataset = DiagramsDataset(test_diagrams, test_labels)

    torch.save(train_dataset, f"../data/mnist/{name}_train.pt")
    torch.save(test_dataset, f"../data/mnist/{name}_test.pt")


if not os.path.exists('../data'):
    os.makedirs('../data')

if not os.path.exists('../data/mnist'):
    os.makedirs('../data/mnist')


train_dataset = ImagesDataset(X_train, y_train)
test_dataset = ImagesDataset(X_test, y_test)

torch.save(train_dataset, f"../data/mnist/images_train.pt")
torch.save(test_dataset, f"../data/mnist/images_test.pt")


### Porus

if not os.path.exists('../data/porus'):
    os.makedirs('../data/porus')

W = 300
sigma1 = 4
sigma2 = 2
t = 0.01

def generate(N, S, W=300, sigma1=4, sigma2=2, t=0.01, bins=64):

    z = np.zeros((N, S, 2))
    for n in range(N):
        z[n, 0] = np.random.uniform(0, W, size=(2))
        for s in range(S-1):
            d_1 = np.random.normal(0, sigma1)
            d_2 = np.random.normal(0, sigma1)
            z[n, s+1, 0] = (z[n, s, 0] + d_1) % W
            z[n, s+1, 1] = (z[n, s, 1] + d_2) % W

    z_r = z.reshape(N*S, 2)
    H, _, _ = np.histogram2d(z_r[:,0], z_r[:,1], bins=bins)
    
    G = gaussian_filter(H, sigma2)
    G[G < t] = 0
    
    return G

count = 10000
classes_count = 2

images = np.zeros((classes_count * count, 64, 64))

# class A
N = 100
S = 30

for n in tqdm(range(count)):
    images[n] = generate(N, S)
    
# class B
N = 250
S = 10

for n in tqdm(range(count)):
    images[n+count] = generate(N, S)

def push_diagrams(diagrams, name):
    labels = [0 for _ in range(count)] + [1 for _ in range(count)]
    pairs = list(zip(diagrams, labels))

    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

    train_diagrams, train_labels = zip(*train_pairs)
    test_diagrams, test_labels = zip(*test_pairs)

    train_dataset = DiagramsDataset(train_diagrams, train_labels)
    test_dataset = DiagramsDataset(test_diagrams, test_labels)

    torch.save(train_dataset, f"../data/porus/{name}_train.pt")
    torch.save(test_dataset, f"../data/porus/{name}_test.pt")

labels = [0 for _ in range(count)] + [1 for _ in range(count)]
pairs = list(zip(images, labels))

train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

train_images, train_labels = zip(*train_pairs)
test_images, test_labels = zip(*test_pairs)

train_dataset = ImagesDataset(train_images, train_labels)
test_dataset = ImagesDataset(test_images, test_labels)

torch.save(train_dataset, f"../data/porus/images_train.pt")
torch.save(test_dataset, f"../data/porus/images_test.pt")


### CIFAR10

if not os.path.exists('../data/cifar10'):
    os.makedirs('../data/cifar10')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
cifar_train = torchvision.datasets.CIFAR10('../data/cifar10/base_cifar10_train_data', train=True, download=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10('../data/cifar10/base_cifar10_test_data', train=False, download=True, transform=transform)
X_train = cifar_train.data / 255
X_test = cifar_test.data / 255
y_train = np.array(cifar_train.targets)
y_test = np.array(cifar_test.targets)

# average on last channel
X_train = np.mean(X_train, axis=-1)
X_test = np.mean(X_test, axis=-1)

def push_diagrams_cifar(train_diagrams, test_diagrams, train_labels, test_labels, name):
    train_dataset = DiagramsDataset(train_diagrams, train_labels)
    test_dataset = DiagramsDataset(test_diagrams, test_labels)

    torch.save(train_dataset, f"../data/cifar10/{name}_train.pt")
    torch.save(test_dataset, f"../data/cifar10/{name}_test.pt")


train_dataset = ImagesDataset(X_train, y_train)
test_dataset = ImagesDataset(X_test, y_test)

torch.save(train_dataset, f"../data/cifar10/images_train.pt")
torch.save(test_dataset, f"../data/cifar10/images_test.pt")

