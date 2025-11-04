# ============================================================
#
# Handles dataset loading, preprocessing, and train–validation
# splitting for all supported image classification benchmarks.
# Provides standardized access to MNIST, Fashion-MNIST, and
# CIFAR-10 datasets with consistent transformations and splits
# for reproducibility across experiments.
#
# ============================================================

from torchvision import datasets
from torch.utils.data import random_split

def load_dataset(name: str, root: str, download: bool, transform):
    if name.lower() == "mnist":
        ds = datasets.MNIST(root, train=True, download=download, transform=transform)
        ds_test = datasets.MNIST(root, train=False, download=download, transform=transform)
    elif name.lower() in ["fashionmnist", "fashion_mnist"]:
        ds = datasets.FashionMNIST(root, train=True, download=download, transform=transform)
        ds_test = datasets.FashionMNIST(root, train=False, download=download, transform=transform)
    elif name.lower() == "cifar10":
        ds = datasets.CIFAR10(root, train=True, download=download, transform=transform)
        ds_test = datasets.CIFAR10(root, train=False, download=download, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return ds, ds_test

def split_train(ds, split_ratio: float = 0.8):
    n = len(ds)
    n_train = int(n * split_ratio)
    n_val = n - n_train
    return random_split(ds, [n_train, n_val])
