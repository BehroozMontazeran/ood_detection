"""
Defines the Datasets used in the
"""

import io
import os
import pickle
import warnings
import zipfile
from abc import ABC, abstractmethod
from io import BytesIO
from os import listdir, path
from pathlib import Path
import pandas as pd
import random
import shutil

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from utilities.routes import DATAROOT

# All greyscale datasets are scaled from [0, 255] to [0, 1]
# All color datasets are scaled from [0, 255] to [-0.5, 0.5]


# class DatasetWrapper:
#     """
#     Primarily associates a fixed name with each dataset so that it can be used in the command line.
#     """
#     name = NotImplementedError()
#     image_shape = NotImplementedError()
#     num_classes = NotImplementedError()

#     @staticmethod
#     def get_train(dataroot=DATAROOT):
#         """Returns the training dataset."""
#         raise NotImplementedError()

#     @staticmethod
#     def get_test(dataroot=DATAROOT):
#         """Returns the test dataset."""
#         raise NotImplementedError()

#     @classmethod
#     def get_all(cls, dataroot=DATAROOT):
#         """Returns a tuple of data used for the dataset (for backwards compatibility with Glow code)."""
#         return cls.image_shape, cls.num_classes, cls.get_train(dataroot), cls.get_test(dataroot)


N_BITS = 8


def preprocess(x):
    """This is the preprocessing used in the Glow code"""
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2**N_BITS
    if N_BITS < 8:
        x = torch.floor(x / 2 ** (8 - N_BITS))
    x = x / n_bins - 0.5

    return x

def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2 ** N_BITS
    return torch.clamp(x, 0, 255).byte()


def one_hot_encode(target):
    """
    One hot encode with fixed 10 classes
    Args: target           - the target labels to one-hot encode
    Retn: one_hot_encoding - the OHE of this tensor
    """
    num_classes = 10
    one_hot_encoding = F.one_hot(torch.tensor(target), num_classes)

    return one_hot_encoding

class DatasetWrapper(ABC):
    """
    Base class for dataset wrappers, associating a fixed name, image shape, and number of classes with each dataset.
    Requires subclasses to implement dataset-specific methods.
    """
    name: str
    image_shape: tuple
    num_classes: int

    @staticmethod
    @abstractmethod
    def get_train(dataroot: str = DATAROOT, transform=None, augment: bool = False, download=True) -> Dataset:
        """Returns the training dataset."""


    @staticmethod
    @abstractmethod
    def get_test(dataroot: str = DATAROOT, transform=None, download=True) -> Dataset:
        """Returns the test dataset."""


    @classmethod
    def get_all(cls, dataroot: str = DATAROOT, transform=None, augment: bool = False, download=True) -> tuple:
        """
        Returns a tuple of dataset info: (image shape, num classes, train dataset, test dataset)
        """
        return (cls.image_shape, cls.num_classes,
                cls.get_train(dataroot, transform, augment, download),
                cls.get_test(dataroot, transform, download))

    @classmethod
    def default_preprocessing(cls) -> list:
        """Defines default preprocessing steps such as scaling or normalization."""
        return [transforms.ToTensor()]

    @classmethod
    def get_augmentations(cls) -> list:
        """Defines default augmentations, to be overridden by subclasses if needed."""
        return []

    @classmethod
    def create_transform(cls, augment: bool, custom_transforms=None) -> transforms.Compose:
        """Creates a transformation pipeline based on whether augmentation is needed."""
        # Call the subclass method for augmentations
        augmentations = cls.get_augmentations() if augment else []

        # Create a pipeline with optional augmentations
        transform_steps = augmentations + cls.default_preprocessing()

        # If custom transforms are provided, use those
        if custom_transforms:
            transform_steps = custom_transforms

        return transforms.Compose(transform_steps)


class MNISTWrapper(DatasetWrapper):
    """MNIST dataset wrapper."""

    name = "mnist"
    image_shape = (28, 28, 1)
    num_classes = 10

    @staticmethod
    def get_train(dataroot: str = DATAROOT, transform=None, augment: bool = False, download: bool=True) -> Dataset:
        if transform is None:
            transform = MNISTWrapper.create_transform(augment)
        return datasets.MNIST(dataroot, train=True, download=download, transform=transform)

    @staticmethod
    def get_test(dataroot: str = DATAROOT, transform=None, download: bool=True) -> Dataset:
        if transform is None:
            transform = MNISTWrapper.create_transform(augment=False)
        return datasets.MNIST(dataroot, train=False, download=download, transform=transform)

class FashionMNISTWrapper(DatasetWrapper):
    """ FashionMNIST dataset wrapper """
    name = "fashionmnist"
    image_shape = (28, 28, 1)
    num_classes = 10

    @staticmethod
    def get_train(dataroot: str=DATAROOT, transform=None, augment: bool=False, download: bool=True) -> Dataset:
        if transform is None:
            transform = MNISTWrapper.create_transform(augment)
        return datasets.FashionMNIST(
            dataroot, train=True, download=download, transform=transform
        )

    @staticmethod
    def get_test(dataroot:str=DATAROOT, transform=None, download: bool=True) -> Dataset:
        if transform is None:
            transform = MNISTWrapper.create_transform(augment=False)
        return datasets.FashionMNIST(
            dataroot, train=False, download=download, transform=transform
        )


class CIFAR10Wrapper(DatasetWrapper):
    """CIFAR10 dataset wrapper."""
    name = "cifar10"
    image_shape = (32, 32, 3)
    num_classes = 10
    pixel_range = (0.5, 0.5)

    @staticmethod
    def root(dataroot):
        """Returns the root directory for CIFAR10."""
        return path.join(dataroot, "CIFAR10")

    @staticmethod
    def get_augmentations() -> list:
        """CIFAR10-specific augmentations."""
        return [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]

    @staticmethod
    def default_preprocessing() -> list:
        return [transforms.ToTensor(), preprocess]

    @staticmethod
    def get_train(dataroot: str = DATAROOT, transform=None, augment: bool = True, download: bool=True) -> Dataset:
        if transform is None:
            transform = CIFAR10Wrapper.create_transform(augment)
        return datasets.CIFAR10(
            CIFAR10Wrapper.root(dataroot),
            train=True,
            transform=transform,
            target_transform=one_hot_encode,
            download=download,
        )


    @staticmethod
    def get_test(dataroot: str = DATAROOT, transform=None, download: bool=True) -> Dataset:
        if transform is None:
            transform = CIFAR10Wrapper.create_transform(augment=False)
        return datasets.CIFAR10(
            CIFAR10Wrapper.root(dataroot),
            train=False,
            transform=transform,
            target_transform=one_hot_encode,
            download=download,)
        


class SVHNWrapper(DatasetWrapper):
    """SVHN dataset wrapper."""
    name = "svhn"
    image_shape = (32, 32, 3)
    num_classes = 10

    @staticmethod
    def root(dataroot):
        """Returns the root directory for SVHN."""
        return path.join(dataroot, "SVHN")

    @staticmethod
    def get_augmentations() -> list:
        """SVHN-specific augmentations."""
        return [transforms.RandomAffine(0, translate=(0.1, 0.1))]

    @staticmethod
    def default_preprocessing() -> list:
        return [transforms.ToTensor(), preprocess]

    @staticmethod
    def get_train(dataroot: str = DATAROOT, transform=None, augment: bool = True, download: bool=True) -> Dataset:
        if transform is None:
            transform = SVHNWrapper.create_transform(augment)
        return datasets.SVHN(
            SVHNWrapper.root(dataroot),
            split="train",
            transform=transform,
            target_transform=one_hot_encode,
            download=download,
        )
    @staticmethod
    def get_test(dataroot: str = DATAROOT, transform=None, download: bool=True) -> Dataset:
        if transform is None:
            transform = SVHNWrapper.create_transform(augment=False)
        return datasets.SVHN(
            SVHNWrapper.root(dataroot),
            split="test",
            transform=transform,
            target_transform=one_hot_encode,
            download=download,
        )

class GTSRBWrapper(DatasetWrapper):
    """For the German Traffic Sign Recognition Benchmark (Houben et al. 2013)"""
    name = "gtsrb"
    image_shape = (32, 32, 3)
    num_classes = 40
    pixel_range = (-0.5, 0.5)

    @staticmethod
    def root(dataroot):
        """Returns the root directory for GTSRB"""
        return path.join(dataroot, "GTSRB")

    @staticmethod
    def get_augmentations() -> list:
        """GTSRB-specific augmentations."""
        return [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    
    @staticmethod
    def default_preprocessing() -> list:
        return [transforms.ToTensor(), transforms.Resize((32, 32)), preprocess]
   
    @staticmethod
    def get_train(dataroot: str=DATAROOT, transform=None, augment=True, download: bool=True) -> Dataset:
        # if augment:
        #     transformations = [
        #         transforms.RandomAffine(0, translate=(0.1, 0.1)),
        #         transforms.RandomHorizontalFlip(),
        #     ]
        # else:
        #     transformations = []

        # if transform is  None:
        #     transformations.extend([transforms.ToTensor(), transforms.Resize((32, 32)), preprocess])
        #     transform = transforms.Compose(transformations)
        if transform is None:
            transform = GTSRBWrapper.create_transform(augment)

        return datasets.GTSRB(
            GTSRBWrapper.root(dataroot),
            split="train",
            transform=transform,
            download=download,
        )

    @staticmethod
    def get_test(dataroot: str=DATAROOT, transform=None, download: bool=True) -> Dataset:
        if transform is None:
            transform = GTSRBWrapper.create_transform(augment=False)

        return datasets.GTSRB(
            GTSRBWrapper.root(dataroot),
            split="test",
            transform=transform,
            download=download,
        )
class OmniglotWrapper(DatasetWrapper):
    """ Omniglot dataset wrapper """
    name = "omniglot"
    image_shape = (28, 28, 1)
    num_classes = 10

    # scaling_transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Resize((image_shape[0], image_shape[1]))]
    # )
    @staticmethod
    def default_preprocessing() -> list:
        return [transforms.ToTensor(), transforms.Resize((OmniglotWrapper.image_shape[0], OmniglotWrapper.image_shape[1]))]

    @staticmethod
    def get_train(dataroot: str=DATAROOT, transform=None, augment: bool=False, download: bool=True) -> Dataset:
        if transform is None:
            transform = OmniglotWrapper.create_transform(augment)
        return datasets.Omniglot(
            dataroot, background=True, download=download, transform=transform
        )

    @staticmethod
    def get_test(dataroot: str=DATAROOT, transform=None, download: bool=True) -> Dataset:
        if transform is None:
            transform = OmniglotWrapper.create_transform(augment=False)
        return datasets.Omniglot(
            dataroot, background=False, download=download, transform=transform
        )
# TODO check if Lambda is working correctly, unless use a function 
class FlippedOmniglotWrapper(OmniglotWrapper):
    """ Flipped Omniglot dataset wrapper """
    name = "flipped_omniglot"

    # scaling_transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         flip,
    #         transforms.Resize((OmniglotWrapper.image_shape[0], OmniglotWrapper.image_shape[1])),
    #     ]
    # )
    @staticmethod
    def default_preprocessing() -> list:
        return [transforms.ToTensor(), transforms.Lambda(lambda x: 1 - x), transforms.Resize((OmniglotWrapper.image_shape[0], OmniglotWrapper.image_shape[1]))]

    @staticmethod
    def get_train(dataroot: str=DATAROOT, transform=None, augment: bool= False, download: bool=True) -> Dataset:
        if transform is None:
            transform = FlippedOmniglotWrapper.create_transform(augment)
        return datasets.Omniglot(
            dataroot, background=True, download=download, transform=transform
        )

    @staticmethod
    def get_test(dataroot: str=DATAROOT, transform=None, download: bool=True) -> Dataset:
        if transform is None:
            transform = FlippedOmniglotWrapper.create_transform(augment=False)
        return datasets.Omniglot(
            dataroot, background=False, download=download, transform=transform
        )

def mnist_scaling(x):
    """ Scales MNIST images from [0, 1] to [-0.5, 0.5] """
    return x - 0.5

# class MNIST_Wrapper(DatasetWrapper):

#     name = "MNIST"
#     image_shape = (28, 28, 1)
#     num_classes = 10

#     @staticmethod
#     def get_train(dataroot=DATAROOT):
#         return datasets.MNIST(
#             dataroot, train=True, download=True, transform=transforms.ToTensor()
#         )

#     @staticmethod
#     def get_test(dataroot=DATAROOT):
#         return datasets.MNIST(
#             dataroot, train=False, download=True, transform=transforms.ToTensor()
#         )
# class MNISTWrapper(DatasetWrapper):
#     """ MNIST dataset wrapper """

#     name = "MNIST"
#     image_shape = (28, 28, 1)
#     num_classes = 10

#     @staticmethod
#     def get_train(dataroot: str=DATAROOT, transform=None) -> torch.utils.data.Dataset:
#         if transform is None:
#             # transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(mnist_scaling)])
#             transform = transforms.ToTensor()
#         return datasets.MNIST(dataroot, train=True, download=True, transform=transform)


#     @staticmethod
#     def get_test(dataroot: str=DATAROOT, transform=None) -> torch.utils.data.Dataset:
#         if transform is None:
#             # transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(mnist_scaling)])
#             transform = transforms.ToTensor()
#         return datasets.MNIST(dataroot, train=False, download=True, transform=transform)


def get_mnist(dataroot):
    """ Returns the MNIST dataset """
    warnings.warn("DEPRECATED. Use MNIST_Wrapper.get_all instead.", DeprecationWarning)
    image_shape = (28, 28, 1)

    num_classes = 10

    train_dataset = datasets.MNIST(
        DATAROOT, train=True, download=True, transform=transforms.ToTensor()
    )
    # transform=transforms.Compose([transforms.ToTensor(), mnist_scaling]))

    test_dataset = datasets.MNIST(
        DATAROOT, train=False, download=True, transform=transforms.ToTensor()
    )
    # transform=transforms.Compose([transforms.ToTensor(), mnist_scaling]))

    return image_shape, num_classes, train_dataset, test_dataset


# class FashionMNISTWrapper(DatasetWrapper):
#     """ FashionMNIST dataset wrapper """
#     name = "FashionMNIST"
#     image_shape = (28, 28, 1)
#     num_classes = 10

#     @staticmethod
#     def get_train(dataroot=DATAROOT):
#         return datasets.FashionMNIST(
#             dataroot, train=True, download=True, transform=transforms.ToTensor()
#         )

#     @staticmethod
#     def get_test(dataroot=DATAROOT):
#         return datasets.FashionMNIST(
#             dataroot, train=False, download=True, transform=transforms.ToTensor()
#         )

def get_fashionmnist(dataroot):
    """ Returns the FashionMNIST dataset """
    warnings.warn("DEPRECATED. Use FashionMNIST_Wrapper.get_all instead.", DeprecationWarning)

    image_shape = (28, 28, 1)

    num_classes = 10

    train_dataset = datasets.FashionMNIST(
        DATAROOT, train=True, download=True, transform=transforms.ToTensor()
    )
    # transform=transforms.Compose([transforms.ToTensor(), mnist_scaling]))

    test_dataset = datasets.FashionMNIST(
        DATAROOT, train=False, download=True, transform=transforms.ToTensor()
    )
    # transform=transforms.Compose([transforms.ToTensor(), mnist_scaling]))

    return image_shape, num_classes, train_dataset, test_dataset


# class OmniglotWrapper(DatasetWrapper):
#     """ Omniglot dataset wrapper """
#     name = "Omniglot"
#     image_shape = (28, 28, 1)
#     num_classes = 10

#     scaling_transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Resize((image_shape[0], image_shape[1]))]
#     )

#     @staticmethod
#     def get_train(dataroot=DATAROOT):
#         return datasets.Omniglot(
#             dataroot, background=True, download=True, transform=OmniglotWrapper.scaling_transform
#         )

#     @staticmethod
#     def get_test(dataroot=DATAROOT):
#         return datasets.Omniglot(
#             dataroot, background=False, download=True, transform=OmniglotWrapper.scaling_transform
#         )

def flip(x):
    """ Flip the image """
    return 1 - x





def get_omniglot(dataroot):
    """ Returns the Omniglot dataset """
    warnings.warn("DEPRECATED. Use Omniglot_Wrapper.get_all instead.", DeprecationWarning)

    image_shape = (28, 28, 1)

    num_classes = 10

    scaling_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((image_shape[0], image_shape[1]))]
    )

    train_dataset = datasets.Omniglot(
        DATAROOT, background=True, download=True, transform=scaling_transform
    )

    test_dataset = datasets.Omniglot(
        DATAROOT, background=False, download=True, transform=scaling_transform
    )

    return image_shape, num_classes, train_dataset, test_dataset


def get_flipped_omniglot(dataroot):
    """ Returns the Flipped Omniglot dataset """
    warnings.warn("DEPRECATED. Use FlippedOmniglotWrapper.get_all instead.", DeprecationWarning)

    # def flip(x):
    #     """ Flips the image """
    #     return 1 - x

    image_shape = (28, 28, 1)

    num_classes = 10

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            flip,
            transforms.Resize((image_shape[0], image_shape[1])),
        ]
    )

    train_dataset = datasets.Omniglot(
        DATAROOT, background=True, download=True, transform=transform
    )

    test_dataset = datasets.Omniglot(
        DATAROOT, background=False, download=True, transform=transform
    )

    return image_shape, num_classes, train_dataset, test_dataset


# class CIFAR10Wrapper(DatasetWrapper):
#     """ CIFAR10 dataset wrapper """
#     name = "cifar10"
#     image_shape = (32, 32, 3)
#     num_classes = 10
#     pixel_range = (0.5, 0.5)

#     @staticmethod
#     def root(dataroot):
#         """Returns the root directory for CIFAR10"""
#         return path.join(dataroot, "CIFAR10")

#     @staticmethod
#     def get_train(dataroot=DATAROOT, augment=True):
#         if augment:
#             transformations = [
#                 transforms.RandomAffine(0, translate=(0.1, 0.1)),
#                 transforms.RandomHorizontalFlip(),
#             ]
#         else:
#             transformations = []

#         transformations.extend([transforms.ToTensor(), preprocess])
#         train_transform = transforms.Compose(transformations)

#         train_dataset = datasets.CIFAR10(
#             CIFAR10Wrapper.root(dataroot),
#             train=True,
#             transform=train_transform,
#             target_transform=one_hot_encode,
#             download=True,
#         )

#         return train_dataset

#     @staticmethod
#     def get_test(dataroot=DATAROOT):

#         test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

#         return datasets.CIFAR10(
#             CIFAR10Wrapper.root(dataroot),
#             train=False,
#             transform=test_transform,
#             target_transform=one_hot_encode,
#             download=True,
#         )
# class CIFAR10Wrapper(DatasetWrapper):
#     """ CIFAR10 dataset wrapper """
#     name = "cifar10"
#     image_shape = (32, 32, 3)
#     num_classes = 10
#     pixel_range = (0.5, 0.5)

#     @staticmethod
#     def root(dataroot):
#         """Returns the root directory for CIFAR10"""
#         return path.join(dataroot, "CIFAR10")

#     @staticmethod
#     def get_train(dataroot: str=DATAROOT, transform=None, augment=True) -> torch.utils.data.Dataset:
#         if augment:
#             transformations = [
#                 transforms.RandomAffine(0, translate=(0.1, 0.1)),
#                 transforms.RandomHorizontalFlip(),
#             ]
#         else:
#             transformations = []

        
#         transform = transforms.Compose(transformations.extend([transforms.ToTensor(), preprocess]))

#         train_dataset = datasets.CIFAR10(
#             CIFAR10Wrapper.root(dataroot),
#             train=True,
#             transform=transform,
#             target_transform=one_hot_encode,
#             download=True,
#         )

#         return train_dataset

#     @staticmethod
#     def get_test(dataroot: str=DATAROOT, transform=None) -> torch.utils.data.Dataset:

#         if transform is None:
#             transform = transforms.Compose([transforms.ToTensor(), preprocess])

#         return datasets.CIFAR10(
#             CIFAR10Wrapper.root(dataroot),
#             train=False,
#             transform=transform,
#             target_transform=one_hot_encode,
#             download=True,
#         )

# class GTSRBWrapper(DatasetWrapper):
#     """For the German Traffic Sign Recognition Benchmark (Houben et al. 2013)"""
#     name = "gtsrb"
#     image_shape = (32, 32, 3)
#     num_classes = 40
#     pixel_range = (-0.5, 0.5)

#     @staticmethod
#     def root(dataroot):
#         """Returns the root directory for GTSRB"""
#         return path.join(dataroot, "GTSRB")

#     @staticmethod
#     def get_train(dataroot=DATAROOT, augment=True):
#         if augment:
#             transformations = [
#                 transforms.RandomAffine(0, translate=(0.1, 0.1)),
#                 transforms.RandomHorizontalFlip(),
#             ]
#         else:
#             transformations = []

#         transformations.extend([transforms.ToTensor(), transforms.Resize((32, 32)), preprocess])
#         train_transform = transforms.Compose(transformations)

#         train_dataset = datasets.GTSRB(
#             GTSRBWrapper.root(dataroot),
#             split="train",
#             transform=train_transform,
#             download=True,
#         )

#         return train_dataset

#     @staticmethod
#     def get_test(dataroot=DATAROOT):

#         test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32)), preprocess])

#         return datasets.GTSRB(
#             GTSRBWrapper.root(dataroot),
#             split="test",
#             transform=test_transform,
#             download=True,
#         )

def get_cifar10(augment, dataroot, download):
    """ Returns the CIFAR10 dataset """
    warnings.warn("DEPRECATED. Use CIFAR10_Wrapper.get_all instead.", DeprecationWarning)

    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    path = Path(dataroot) / "data" / "CIFAR10"
    train_dataset = datasets.CIFAR10(
        path,
        train=True,
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.CIFAR10(
        path,
        train=False,
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset


# class SVHNWrapper(DatasetWrapper):
#     """For the Street View House Numbers dataset (Netzer et al. 2011)"""
#     name = "svhn"
#     image_shape = (32, 32, 3)
#     num_classes = 10

#     @staticmethod
#     def root(dataroot):
#         """Returns the root directory for SVHN"""
#         return path.join(dataroot, "SVHN")

#     @staticmethod
#     def get_train(dataroot: str=DATAROOT, transform=None, augment=True) -> torch.utils.data.Dataset:
#         if augment:
#             transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
#         else:
#             transformations = []
    
#         transformations.extend([transforms.ToTensor(), preprocess])
#         transform = transforms.Compose(transformations)

#         return datasets.SVHN(
#             SVHNWrapper.root(dataroot),
#             split="train",
#             transform=transform,
#             target_transform=one_hot_encode,
#             download=True,
#         )

#     @staticmethod
#     def get_test(dataroot: str=DATAROOT, transform=None) -> torch.utils.data.Dataset:
#         transform = transforms.Compose([transforms.ToTensor(), preprocess])

#         return datasets.SVHN(
#             SVHNWrapper.root(dataroot),
#             split="test",
#             transform=transform,
#             target_transform=one_hot_encode,
#             download=True,
#         )


def get_svhn(augment, dataroot, download):
    """ Returns the SVHN dataset """
    warnings.warn("DEPRECATED. Use SVHN_Wrapper.get_all instead.", DeprecationWarning)

    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    path = Path(dataroot) / "data" / "SVHN"

    print(f"dataroot: {dataroot}")
    print(f"path: {path}")

    train_dataset = datasets.SVHN(
        path,
        split="train",
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.SVHN(
        path,
        split="test",
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset

# TODO: Implement the following datasets based on new changes



class ImageNet32Wrapper(DatasetWrapper):
    """ImageNet32 dataset wrapper."""

    name = "imagenet32"
    image_shape = (32, 32, 3)
    num_classes = 1000

    @staticmethod
    def root(dataroot):
        """Returns the root directory for IMAGENET32."""
        return os.path.join(dataroot, "IMAGENET32")

    @staticmethod
    def get_augmentations() -> list:
        """IMAGENET32-specific augmentations."""
        return [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]

    @staticmethod
    def default_preprocessing() -> list:
        return [transforms.ToTensor(), preprocess]

    @staticmethod
    def extract_zip(zip_path: str, extract_to: str) -> None:
        """Extracts a zip file to a specified directory."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    @staticmethod
    def load_batch_file(file_path: str) -> tuple:
        """Loads a batch file and returns images and labels."""
        with open(file_path, 'rb') as f:
            batch = pickle.load(f)
            images = batch['data']
            labels = batch['labels']


        # Transform images from raw to PIL format
        images = [Image.fromarray(img.reshape(3, 32, 32).transpose(1, 2, 0)) for img in images]  # Convert to (H, W, C)
        return images, labels

    @staticmethod
    def load_images_from_batches(folder: str, transform=None) -> tuple:
        """Loads images and labels from all batch files in a folder."""
        all_images = []
        all_labels = []
        
        # Iterate over files in the provided folder
        for filename in listdir(folder):
            file_path = path.join(folder, filename)

            if path.isfile(file_path):
                # Handle training batches
                if filename.startswith("train_data_batch_"):
                    try:
                        images, labels = ImageNet32Wrapper.load_batch_file(file_path)
                        if transform:
                            images = [transform(img) for img in images]
                        all_images.extend(images)
                        all_labels.extend(labels)
                    except Exception as e:
                        print(f"Error loading batch file {file_path}: {e}")

                # Handle validation batch
                elif filename == "val_data":
                    try:
                        images, labels = ImageNet32Wrapper.load_batch_file(file_path)
                        if transform:
                            images = [transform(img) for img in images]
                        all_images.extend(images)
                        all_labels.extend(labels)
                    except Exception as e:
                        print(f"Error loading validation file {file_path}: {e}")

        return all_images, all_labels


    @staticmethod
    def create_lmdb_from_batches(lmdb_path: str, batch_folder: str, transform=None):
        """Creates an LMDB database from ImageNet32 batch files."""
        env = lmdb.open(lmdb_path, map_size=int(1e10))  # Adjust map size as necessary

        with env.begin(write=True) as txn:
            # Load images and labels from the extracted batch files
            images, labels = ImageNet32Wrapper.load_images_from_batches(batch_folder, transform)
            for idx, (img, label) in tqdm(enumerate(zip(images, labels)), desc="Creating LMDB", total=len(images)):
                # Serialize each image and label as bytes
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                img_bytes = buffer.getvalue()
                key = f"{idx:08}".encode("ascii")
                value = pickle.dumps((img_bytes, label))
                if not txn.put(key, value):
                    print(f"Warning: Failed to write entry {idx}")


        env.close()
        print("LMDB database created successfully at:", lmdb_path)
        print(f"Total images added to LMDB: {idx+1}")

    def __init__(self, lmdb_path: str, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        # Calculate the number of samples
        with self.env.begin() as txn:
            self.length = txn.stat()["entries"]
            print(f"Total entries in LMDB: {self.length}")

    def __getitem__(self, index):
        if index >= self.length:
            # Handle the out-of-bounds case by returning None or a placeholder.
            print(f"{index} is out of bounds") # Debugging Print
            return None, None
            # continue
        key = f"{index:08}".encode("ascii")
        with self.env.begin(write=False) as txn:
            data = txn.get(key)
            if data is None:
                raise KeyError(f"Entry not found in LMDB for key: {key}")

            img_bytes, label = pickle.loads(data)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            if self.transform:
                img = self.transform(img)

        return img, label

    def __len__(self):
        return self.length
    # TODO: Implement batch loading for ImageNet32 for both training and validation sets
    @staticmethod
    def get_train(dataroot: str = DATAROOT, transform=None, augment: bool = True, download: bool = False) -> Dataset:
        """Returns the training dataset."""
        train_zip_path = path.join(ImageNet32Wrapper.root(dataroot), 'Imagenet32_train.zip')
        extract_path = path.join(ImageNet32Wrapper.root(dataroot), 'train_batches')
        
        # Extract the training data if it hasn't been extracted yet
        if not path.exists(extract_path):
            ImageNet32Wrapper.extract_zip(train_zip_path, extract_path)


        lmdb_path = os.path.join(ImageNet32Wrapper.root(dataroot), 'train.lmdb')
        batch_folder = os.path.join(ImageNet32Wrapper.root(dataroot), 'train_batches')

        # If LMDB doesn't exist, create it
        if not os.path.exists(lmdb_path):
            ImageNet32Wrapper.create_lmdb_from_batches(lmdb_path, batch_folder, transform)

        # Define transformations
        if transform is None:
            transform = ImageNet32Wrapper.create_transform(augment)

        dataset = ImageNet32Wrapper(lmdb_path, transform)
        images, labels = [], [] 
        for i in tqdm(range(len(dataset)), desc="Loading training data"):
            img, label = dataset[i]
            images.append(img)
            labels.append(label)

        return torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))

    @staticmethod
    def get_test(dataroot: str = DATAROOT, transform=None, download: bool = False) -> Dataset:
        """Returns the validation dataset."""

        val_zip_path = path.join(ImageNet32Wrapper.root(dataroot), 'Imagenet32_val.zip')
        extract_path = path.join(ImageNet32Wrapper.root(dataroot), 'val_batches')

        # Extract the validation data if it hasn't been extracted yet
        if not path.exists(extract_path):
            ImageNet32Wrapper.extract_zip(val_zip_path, extract_path)


        lmdb_path = os.path.join(ImageNet32Wrapper.root(dataroot), 'val.lmdb')
        batch_folder = os.path.join(ImageNet32Wrapper.root(dataroot), 'val_batches')

        # If LMDB doesn't exist, create it
        if not os.path.exists(lmdb_path):
            ImageNet32Wrapper.create_lmdb_from_batches(lmdb_path, batch_folder, transform)

        # Define transformations
        if transform is None:
            transform = ImageNet32Wrapper.create_transform(augment=False)

        dataset = ImageNet32Wrapper(lmdb_path, transform)
        images, labels = [], []
        for i in tqdm(range(len(dataset)), desc="Loading validation data"):
            img, label = dataset[i]
            images.append(img)
            labels.append(label)

        return torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))

# class ImageNet32Wrapper(DatasetWrapper):
#     """ImageNet32 dataset wrapper."""
    
#     name = "imagenet32"
#     image_shape = (32, 32, 3)
#     num_classes = 1000


#     @staticmethod
#     def root(dataroot):
#         """Returns the root directory for IMAGENET32."""
#         return path.join(dataroot, "IMAGENET32")

#     @staticmethod
#     def get_augmentations() -> list:
#         """IMAGENET32-specific augmentations."""
#         return [
#             transforms.RandomAffine(0, translate=(0.1, 0.1)),
#             transforms.RandomHorizontalFlip(),
#         ]

#     @staticmethod
#     def default_preprocessing() -> list:
#         return [transforms.ToTensor(), preprocess]

#     @staticmethod
#     def extract_zip(zip_path: str, extract_to: str) -> None:
#         """Extracts a zip file to a specified directory."""
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_to)

#     @staticmethod
#     def load_batch_file(file_path: str) -> tuple:
#         """Loads a batch file and returns images and labels."""
#         with open(file_path, 'rb') as f:
#             batch = pickle.load(f)
#             images = batch['data']
#             labels = batch['labels']
        
#         # Transform images from raw to PIL format
#         images = [Image.fromarray(img.reshape(3, 32, 32).transpose(1, 2, 0)) for img in images]  # Convert to (H, W, C)
#         return images, labels

#     @staticmethod
#     def load_images_from_batches(folder: str, transform=None) -> tuple:
#         """Loads images and labels from all batch files in a folder."""
#         all_images = []
#         all_labels = []
        
#         # Iterate over files in the provided folder
#         for filename in listdir(folder):
#             file_path = path.join(folder, filename)

#             if path.isfile(file_path):
#                 # Handle training batches
#                 if filename.startswith("train_data_batch_"):
#                     try:
#                         images, labels = ImageNet32Wrapper.load_batch_file(file_path)
#                         if transform:
#                             images = [transform(img) for img in images]
#                         all_images.extend(images)
#                         all_labels.extend(labels)
#                     except Exception as e:
#                         print(f"Error loading batch file {file_path}: {e}")

#                 # Handle validation batch
#                 elif filename == "val_data":
#                     try:
#                         images, labels = ImageNet32Wrapper.load_batch_file(file_path)
#                         if transform:
#                             images = [transform(img) for img in images]
#                         all_images.extend(images)
#                         all_labels.extend(labels)
#                     except Exception as e:
#                         print(f"Error loading validation file {file_path}: {e}")

#         return all_images, all_labels

#     @staticmethod
#     def get_train(dataroot: str = DATAROOT, transform=None, augment: bool = False, download: bool = False) -> torch.utils.data.Dataset:
#         """Returns the training dataset."""
#         train_zip_path = path.join(ImageNet32Wrapper.root(dataroot), 'Imagenet32_train.zip')
#         extract_path = path.join(ImageNet32Wrapper.root(dataroot), 'train')
        
#         # Extract the training data if it hasn't been extracted yet
#         if download and not path.exists(extract_path):
#             ImageNet32Wrapper.extract_zip(train_zip_path, extract_path)
        
#         if transform is None:
#             transform = ImageNet32Wrapper.create_transform(augment)

#         # Load images and labels from the extracted batch files
#         train_images, train_labels = ImageNet32Wrapper.load_images_from_batches(extract_path, transform)

#         # Convert lists to tensors
#         train_images_tensor = torch.stack(train_images[:60000])
#         train_labels_tensor = torch.tensor(train_labels[:60000])

#         return torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)

#     @staticmethod
#     def get_test(dataroot: str = DATAROOT, transform=None, download: bool = False) -> torch.utils.data.Dataset:
#         """Returns the validation dataset."""
#         val_zip_path = path.join(ImageNet32Wrapper.root(dataroot), 'Imagenet32_val.zip')
#         extract_path = path.join(ImageNet32Wrapper.root(dataroot), 'val')

#         # Extract the validation data if it hasn't been extracted yet
#         if download and not path.exists(extract_path):
#             ImageNet32Wrapper.extract_zip(val_zip_path, extract_path)
        
#         if transform is None:
#             transform = ImageNet32Wrapper.create_transform(augment=False)
        
#         # Load images and labels from the extracted batch files
#         val_images, val_labels = ImageNet32Wrapper.load_images_from_batches(extract_path, transform)

#         # Convert lists to tensors
#         val_images_tensor = torch.stack(val_images[:10000])
#         val_labels_tensor = torch.tensor(val_labels[:10000])

#         return torch.utils.data.TensorDataset(val_images_tensor, val_labels_tensor)


class CropCelebA64():
    """This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """

    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + "()"

class CelebAWrapper(DatasetWrapper):
    """CelebA dataset wrapper."""
    name = "celeba"
    image_shape = (32, 32, 3)
    num_classes = 40  # CelebA has 40 binary attribute classes

    def __init__(self, lmdb_path: str, transform=None):
        # self.img_folder = img_folder
        self.transform = transform
        # self.split = split
        # self.split_ratio = split_ratio
        # Load attributes
        # self.attributes = self.load_attributes(attributes_file)

        # Split dataset into train and test
        # self.train_files, self.test_files = self.get_split_files()
        # self.data_files = self.train_files if split == "train" else self.test_files
        self.lmdb_path = lmdb_path
        # # Set the appropriate LMDB path for train/test
        # self.lmdb_path = os.path.join(lmdb_path, f"{split}.lmdb")

        # # Check if LMDB file exists; if not, create it
        # if not os.path.exists(self.lmdb_path):
        #     print(f"{self.lmdb_path} does not exist. Creating LMDB for {split} data...")
        #     self.create_lmdb_from_images(self.lmdb_path, self.img_folder, self.data_files, self.attributes, self.transform)

        # Open LMDB after ensuring it exists
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = txn.stat()["entries"]
            print(f"Total entries in LMDB split): {self.length}")


    @staticmethod
    def root(dataroot):
        """Returns the root directory for CelebA."""
        return os.path.join(dataroot, "CELEBA")

    # @staticmethod
    # def get_augmentations() -> list:
    #     """CelebA-specific augmentations."""
    #     return [
    #         transforms.RandomAffine(0, translate=(0.1, 0.1)),
    #         transforms.RandomHorizontalFlip(p=0.1),
    #     ]
    @staticmethod
    def get_split_files(img_folder: str, attributes: dict, split_ratio: float=0.8):
        """Loads or creates train and test split files and physically splits images."""
        train_split_path = os.path.join(img_folder, "train_imgs")
        test_split_path = os.path.join(img_folder, "test_imgs")

        if os.path.exists(train_split_path) and os.path.exists(test_split_path):
            # If folders already exist, get list of files from them
            print("Loading existing train/test split.")
            train_files = [os.path.join("train_imgs", fname) for fname in os.listdir(train_split_path)]
            test_files = [os.path.join("test_imgs", fname) for fname in os.listdir(test_split_path)]
        else:
            # Create folders for train and test splits
            os.makedirs(train_split_path, exist_ok=True)
            os.makedirs(test_split_path, exist_ok=True)
            
            # Get all file names
            all_files = list(attributes.keys())
            random.shuffle(all_files)  # Shuffle files for random splitting

            # Split the data
            split_index = int(len(all_files) * split_ratio)
            train_files = all_files[:split_index]
            test_files = all_files[split_index:]

            # Move files into their respective directories
            for img_name in tqdm(train_files, desc="Moving train files"):
                source_path = os.path.join(img_folder, img_name)
                target_path = os.path.join(train_split_path, img_name)
                shutil.move(source_path, target_path)  # Move file to train folder

            for img_name in tqdm(test_files, desc="Moving test files"):
                source_path = os.path.join(img_folder, img_name)
                target_path = os.path.join(test_split_path, img_name)
                shutil.move(source_path, target_path)  # Move file to test folder

            # Update paths with new locations
            train_files = [os.path.join("train_imgs", fname) for fname in train_files]
            test_files = [os.path.join("test_imgs", fname) for fname in test_files]

        return train_files, test_files


    @staticmethod
    def default_preprocessing() -> list:
        return [CropCelebA64(),transforms.Resize((32, 32)),
                 transforms.ToTensor(), preprocess]

    @staticmethod
    def extract_zip(zip_path: str, extract_to: str) -> None:
        """Extracts a zip file to a specified directory."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    # def split_data(self, split_ratio=0.8):
    #     """Split images into train and test sets."""
    #     all_files = list(self.attributes.keys())
    #     random.shuffle(all_files)  # Shuffle images for random splitting

    #     split_index = int(len(all_files) * split_ratio)
    #     train_files = all_files[:split_index]
    #     test_files = all_files[split_index:]

    #     return train_files, test_files
    
    @staticmethod
    def load_attributes(file_path: str) -> dict:
        """Loads CelebA attributes from a file."""
        df = pd.read_csv(file_path)
        attributes = {}
        
        # Iterate over each row in the DataFrame to map image_id to its attributes
        for _, row in tqdm(df.iterrows(), desc="Loading CelebA attributes", total=len(df)):
            img_name = row['image_id']
            label = row.iloc[1:].tolist()  # Get all attribute columns as list
            attributes[img_name] = label
        
        return attributes

    @staticmethod
    def create_lmdb_from_images(lmdb_path: str, images_folder: str, data_files: list, attributes: dict, transform=None):
        """Creates an LMDB database from CelebA images."""
        env = lmdb.open(lmdb_path, map_size=int(1e10))
        with env.begin(write=True) as txn:
            for idx, img_name in tqdm(enumerate(data_files), desc=f"Creating LMDB at {lmdb_path}", total=len(data_files)):
                img_path = os.path.join(images_folder, img_name)
                img = Image.open(img_path).convert("RGB")
                if transform:
                    img = transform(img)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                img_bytes = buffer.getvalue()
                label = attributes[path.basename(img_name)]
                key = f"{idx:08}".encode("ascii")
                value = pickle.dumps((img_bytes, label))
                txn.put(key, value)
        env.close()
        print(f"LMDB database created successfully at: {lmdb_path}")
        print(f"Total images added to LMDB: {idx+1}")

    def __getitem__(self, index):
        # if index >= self.length:
        #     return None, None
        key = f"{index:08}".encode("ascii")
        # env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            data = txn.get(key)
            if data is None:
                raise KeyError(f"Entry not found in LMDB for key: {key}")
            img_bytes, label = pickle.loads(data)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, label

    def __len__(self):
        # return len(self.data_files)
        return self.length

    @staticmethod
    def get_train(dataroot: str = DATAROOT, transform=None, augment: bool = True, download: bool = False) -> Dataset:
        """Returns the training dataset."""
        train_zip_path = path.join(CelebAWrapper.root(dataroot), 'archive.zip')
        images_folder = path.join(CelebAWrapper.root(dataroot), 'img')
        attr_file = os.path.join(CelebAWrapper.root(dataroot), "list_attr_celeba.csv")

        if not os.path.exists(images_folder) or not os.path.exists(attr_file):
            CelebAWrapper.extract_zip(train_zip_path, images_folder)

        lmdb_path = os.path.join(CelebAWrapper.root(dataroot), 'train.lmdb')
        if not os.path.exists(lmdb_path):
            attributes = CelebAWrapper.load_attributes(attr_file)
            train_files, _ = CelebAWrapper.get_split_files(images_folder, attributes)
            CelebAWrapper.create_lmdb_from_images(lmdb_path, images_folder, train_files , attributes, transform)

        if transform is None:
            transform = CelebAWrapper.create_transform(augment)

        dataset = CelebAWrapper(lmdb_path, transform)
        images, labels = [], [] 
        for i in tqdm(range(len(dataset)), desc="Loading training data"):
            img, label = dataset[i]
            images.append(img)
            labels.append(label)
        return torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))

    @staticmethod
    def get_test(dataroot: str=DATAROOT, transform=None, download: bool = False) -> Dataset:
        """Returns the validation dataset."""
        val_zip_path = path.join(CelebAWrapper.root(dataroot), 'archive.zip')
        images_folder = path.join(CelebAWrapper.root(dataroot), 'img')
        attr_file = os.path.join(CelebAWrapper.root(dataroot), "list_attr_celeba.csv")

        if not os.path.exists(images_folder) or not os.path.exists(attr_file):
            CelebAWrapper.extract_zip(val_zip_path, images_folder)

        lmdb_path = os.path.join(CelebAWrapper.root(dataroot), 'val.lmdb')
        if not os.path.exists(lmdb_path):
            attributes = CelebAWrapper.load_attributes(attr_file)
            _, test_files = CelebAWrapper.get_split_files(images_folder, attributes)
            CelebAWrapper.create_lmdb_from_images(lmdb_path, images_folder, test_files, attributes, transform)

        if transform is None:
            transform = CelebAWrapper.create_transform(augment=False)

        dataset = CelebAWrapper(lmdb_path, transform)
        images, labels = [], [] 
        for i in tqdm(range(len(dataset)), desc="Loading validation data"):
            img, label = dataset[i]
            images.append(img)
            labels.append(label)
        return torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))





def get_imagenet32(dataroot):
    """ Returns the Imagenet32 dataset """
    warnings.warn("DEPRECATED. Use Imagenet32_Wrapper.get_all instead.", DeprecationWarning)

    image_shape = (32, 32, 3)
    num_classes = None  

    x_train_list = []
    for i in range(1, 11):
        x_train_batch = load_databatch(
            path.join(
                dataroot,
                "data",
                "imagenet32_regular",
                "train_32x32",
                "train_data_batch_" + str(i),
            )
        )
        x_train_list.append(x_train_batch)

    x_train = np.concatenate(x_train_list)
    x_test = load_databatch(
        path.join(dataroot, "data", "imagenet32_regular", "valid_32x32", "val_data")
    )

    
    dummy_train_labels = torch.zeros(len(x_train))
    dummy_test_labels = torch.zeros(len(x_test))

    train_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32), dummy_train_labels
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x_test, dtype=torch.float32), dummy_test_labels
    )

    return image_shape, num_classes, train_dataset, test_dataset


# def get_celeba(dataroot):
#     """ Returns the CelebA dataset """
#     warnings.warn("DEPRECATED. Use Imagenet32_Wrapper.get_all instead.", DeprecationWarning)

#     class CelebaLMBDWrapper:
#         """Creates dummy labels and scales the images to be in the range (-0.5, 0.5)"""
#         def __init__(self, lmdb_dataset):
#             self.lmdb_dataset = lmdb_dataset

#         def __len__(self):
#             return self.lmdb_dataset.__len__()

#         def __getitem__(self, item):
#             sample = self.lmdb_dataset.__getitem__(item)
#             sample = sample - 0.5
#             return sample, torch.zeros(1)

#     image_shape = (32, 32, 3)
#     resize = 32

#     num_classes = None

#     train_transform, valid_transform = _data_transforms_celeba64(resize)
#     train_data = LMDBDataset(
#         root=path.join(dataroot, "data/celeba64_lmdb"),
#         name="celeba64",
#         split="train",
#         transform=train_transform,
#         is_encoded=True,
#     )
#     valid_data = LMDBDataset(
#         root=path.join(dataroot,"data/celeba64_lmdb"),
#         name="celeba64",
#         split="validation",
#         transform=valid_transform,
#         is_encoded=True,
#     )
#     test_data = LMDBDataset(
#         root=path.join(dataroot, "data/celeba64_lmdb"),
#         name="celeba64",
#         split="test",
#         transform=valid_transform,
#         is_encoded=True,
#     )

#     return (
#         image_shape,
#         num_classes,
#         CelebaLMBDWrapper(train_data),
#         CelebaLMBDWrapper(test_data),
#     )


# --------------


def flatten(outer):
    """Flattens a list of lists"""
    return [el for inner in outer for el in inner]


def load_databatch(path, img_size=32):
    """
    As copied from https://patrykchrabaszcz.github.io/Imagenet32/
    """
    d = unpickle_imagenet32(path)
    x = d["data"]  # is already uint8

    img_size2 = img_size * img_size
    x = np.dstack(
        (x[:, :img_size2], x[:, img_size2 : 2 * img_size2], x[:, 2 * img_size2 :])
    )
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(
        0, 3, 1, 2
    )  # (do not transpose, since we want (samples, 32, 32, 3))

    x = x / 256 - 0.5

    return x


def unpickle_imagenet32(file):
    """
    As copied from https://patrykchrabaszcz.github.io/Imagenet32/
    """
    with open(file, "rb") as fo:
        dic = pickle.load(fo)
    return dic


# class CropCelebA64(object):
#     """This class applies cropping for CelebA64. This is a simplified implementation of:
#     https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
#     """

#     def __call__(self, pic):
#         new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
#         return new_pic

#     def __repr__(self):
#         return self.__class__.__name__ + "()"


# def _data_transforms_celeba64(size):
#     train_transform = transforms.Compose(
#         [
#             CropCelebA64(),
#             transforms.Resize(size),
#             # transforms.RandomHorizontalFlip(),  # taken out compared to NVAE --> we don't want data augmentation
#             transforms.ToTensor(),
#         ]
#     )

#     valid_transform = transforms.Compose(
#         [
#             CropCelebA64(),
#             transforms.Resize(size),
#             transforms.ToTensor(),
#         ]
#     )

#     return train_transform, valid_transform


# class CelebaWrapper(DatasetWrapper):
#     """For the CelebA dataset (Liu et al. 2015)"""
#     name = "celeba"
#     image_shape = (32, 32, 3)
#     num_classes = NotImplementedError("currently labels for CelebaA are unimplemented")

#     resize = 32
#     train_transform, valid_transform = _data_transforms_celeba64(resize)

#     @staticmethod
#     def root(dataroot):
#         """Returns the root directory for CelebA"""
#         return path.join(dataroot, "celeba64_lmdb")

#     class CelebaLMBDDataset:
#         """Creates dummy labels and scales the images them to be in the range (-0.5, 0.5)"""
#         def __init__(self, lmdb_dataset):
#             self.lmdb_dataset = lmdb_dataset

#         def __len__(self):
#             return self.lmdb_dataset.__len__()

#         def __getitem__(self, item):
#             sample = self.lmdb_dataset.__getitem__(item)
#             sample = sample - 0.5
#             return sample, torch.zeros(1)

#     @staticmethod
#     def get_train(dataroot=DATAROOT):
#         return CelebaWrapper.CelebaLMBDDataset(
#             LMDBDataset(
#                 root=CelebaWrapper.root(dataroot),
#                 name="celeba64",
#                 split="train",
#                 transform=CelebaWrapper.train_transform,
#                 is_encoded=True,
#             )
#         )

#     @staticmethod
#     def get_test(dataroot=DATAROOT):
#         return CelebaWrapper.CelebaLMBDDataset(
#             LMDBDataset(
#                 root=CelebaWrapper.root(dataroot),
#                 name="celeba64",
#                 split="test",
#                 transform=CelebaWrapper.train_transform,
#                 is_encoded=True,
#             )
#         )


# def num_samples(dataset, split):
#     """Returns the number of samples in the dataset"""
#     if dataset == "celeba":
#         # return 27000 if train else 3000
#         pass
#     elif dataset == "celeba64":
#         if split == "train":
#             return 162770
#         elif split == "validation":
#             return 19867
#         elif split == "test":
#             return 19962
#     else:
#         raise NotImplementedError(f"dataset {dataset} is unknown")


# class LMDBDataset(data.Dataset):
#     """Dataset for LMDB files"""
#     def __init__(self, root, name="", split="train", transform=None, is_encoded=False):
#         self.name = name
#         self.split = split
#         self.transform = transform
#         if self.split in ["train", "validation", "test"]:
#             lmdb_path = path.join(root, f"{self.split}.lmdb")
#         else:
#             print("invalid split param")
#         self.data_lmdb = lmdb.open(
#             lmdb_path,
#             readonly=True,
#             max_readers=1,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )
#         self.is_encoded = is_encoded

#     def __getitem__(self, index):
#         target = [0]
#         with self.data_lmdb.begin(write=False, buffers=True) as txn:
#             data = txn.get(str(index).encode())
#             if self.is_encoded:
#                 img = Image.open(io.BytesIO(data))
#                 img = img.convert("RGB")
#             else:
#                 img = np.asarray(data, dtype=np.uint8)
#                 # assume data is RGB
#                 size = int(np.sqrt(len(img) / 3))
#                 img = np.reshape(img, (size, size, 3))
#                 img = Image.fromarray(img, mode="RGB")

#         if self.transform is not None:
#             img = self.transform(img)

#         return img

#     def __len__(self):
#         return num_samples(self.name, self.split)
