"""
Defines the Datasets used in the
"""

import io
import pickle
import zipfile
import warnings
from abc import ABC, abstractmethod
from os import path, listdir
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

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
        return path.join(dataroot, "IMAGENET32")

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
    def get_train(dataroot: str = DATAROOT, transform=None, augment: bool = False, download: bool = False) -> torch.utils.data.Dataset:
        """Returns the training dataset."""
        train_zip_path = path.join(ImageNet32Wrapper.root(dataroot), 'Imagenet32_train.zip')
        extract_path = path.join(ImageNet32Wrapper.root(dataroot), 'train')
        
        # Extract the training data if it hasn't been extracted yet
        if download and not path.exists(extract_path):
            ImageNet32Wrapper.extract_zip(train_zip_path, extract_path)
        
        if transform is None:
            transform = ImageNet32Wrapper.create_transform(augment)

        # Load images and labels from the extracted batch files
        train_images, train_labels = ImageNet32Wrapper.load_images_from_batches(extract_path, transform)

        # Convert lists to tensors
        train_images_tensor = torch.stack(train_images[:60000])
        train_labels_tensor = torch.tensor(train_labels[:60000])

        return torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)

    @staticmethod
    def get_test(dataroot: str = DATAROOT, transform=None, download: bool = False) -> torch.utils.data.Dataset:
        """Returns the validation dataset."""
        val_zip_path = path.join(ImageNet32Wrapper.root(dataroot), 'Imagenet32_val.zip')
        extract_path = path.join(ImageNet32Wrapper.root(dataroot), 'val')

        # Extract the validation data if it hasn't been extracted yet
        if download and not path.exists(extract_path):
            ImageNet32Wrapper.extract_zip(val_zip_path, extract_path)
        
        if transform is None:
            transform = ImageNet32Wrapper.create_transform(augment=False)
        
        # Load images and labels from the extracted batch files
        val_images, val_labels = ImageNet32Wrapper.load_images_from_batches(extract_path, transform)

        # Convert lists to tensors
        val_images_tensor = torch.stack(val_images[:10000])
        val_labels_tensor = torch.tensor(val_labels[:10000])

        return torch.utils.data.TensorDataset(val_images_tensor, val_labels_tensor)


# class Imagenet32Wrapper(DatasetWrapper):
#     """For the Imagenet32 dataset (Chrabaszcz et al. 2017)"""
#     name = "imagenet32"
#     image_shape = (32, 32, 3)
#     num_classes = NotImplementedError("currently labels for imagenet32 are unimplemented")

#     @staticmethod
#     def get_train(dataroot=DATAROOT, transform=None, augment=False, download=True) -> torch.utils.data.Dataset:
#         x_train_list = []
#         for i in range(1, 11):
#             x_train_batch = load_databatch(
#                 path.join(
#                     dataroot,
#                     "imagenet32_regular",
#                     "train_32x32",
#                     "train_data_batch_" + str(i),
#                 )
#             )
#             x_train_list.append(x_train_batch)

#         x_train = np.concatenate(x_train_list)
#         dummy_train_labels = torch.zeros(len(x_train))

#         return torch.utils.data.TensorDataset(
#             torch.as_tensor(x_train, dtype=torch.float32), dummy_train_labels
#         )

#     @staticmethod
#     def get_test(dataroot=DATAROOT, transform=None, download=True) -> torch.utils.data.Dataset:
#         x_test = load_databatch(
#             path.join(
#                 dataroot,
#                 "imagenet32_regular",
#                 "valid_32x32",
#                 "val_data")
#         )

#         dummy_test_labels = torch.zeros(len(x_test))
#         return torch.utils.data.TensorDataset(
#             torch.as_tensor(x_test, dtype=torch.float32), dummy_test_labels
#         )


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


def get_celeba(dataroot):
    """ Returns the CelebA dataset """
    warnings.warn("DEPRECATED. Use Imagenet32_Wrapper.get_all instead.", DeprecationWarning)

    class CelebaLMBDWrapper:
        """Creates dummy labels and scales the images to be in the range (-0.5, 0.5)"""
        def __init__(self, lmdb_dataset):
            self.lmdb_dataset = lmdb_dataset

        def __len__(self):
            return self.lmdb_dataset.__len__()

        def __getitem__(self, item):
            sample = self.lmdb_dataset.__getitem__(item)
            sample = sample - 0.5
            return sample, torch.zeros(1)

    image_shape = (32, 32, 3)
    resize = 32

    num_classes = None

    train_transform, valid_transform = _data_transforms_celeba64(resize)
    train_data = LMDBDataset(
        root=path.join(dataroot, "data/celeba64_lmdb"),
        name="celeba64",
        split="train",
        transform=train_transform,
        is_encoded=True,
    )
    valid_data = LMDBDataset(
        root=path.join(dataroot,"data/celeba64_lmdb"),
        name="celeba64",
        split="validation",
        transform=valid_transform,
        is_encoded=True,
    )
    test_data = LMDBDataset(
        root=path.join(dataroot, "data/celeba64_lmdb"),
        name="celeba64",
        split="test",
        transform=valid_transform,
        is_encoded=True,
    )

    return (
        image_shape,
        num_classes,
        CelebaLMBDWrapper(train_data),
        CelebaLMBDWrapper(test_data),
    )


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


class CropCelebA64(object):
    """This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """

    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + "()"


def _data_transforms_celeba64(size):
    train_transform = transforms.Compose(
        [
            CropCelebA64(),
            transforms.Resize(size),
            # transforms.RandomHorizontalFlip(),  # taken out compared to NVAE --> we don't want data augmentation
            transforms.ToTensor(),
        ]
    )

    valid_transform = transforms.Compose(
        [
            CropCelebA64(),
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )

    return train_transform, valid_transform


class CelebaWrapper(DatasetWrapper):
    """For the CelebA dataset (Liu et al. 2015)"""
    name = "celeba"
    image_shape = (32, 32, 3)
    num_classes = NotImplementedError("currently labels for CelebaA are unimplemented")

    resize = 32
    train_transform, valid_transform = _data_transforms_celeba64(resize)

    @staticmethod
    def root(dataroot):
        """Returns the root directory for CelebA"""
        return path.join(dataroot, "celeba64_lmdb")

    class CelebaLMBDDataset:
        """Creates dummy labels and scales the images them to be in the range (-0.5, 0.5)"""
        def __init__(self, lmdb_dataset):
            self.lmdb_dataset = lmdb_dataset

        def __len__(self):
            return self.lmdb_dataset.__len__()

        def __getitem__(self, item):
            sample = self.lmdb_dataset.__getitem__(item)
            sample = sample - 0.5
            return sample, torch.zeros(1)

    @staticmethod
    def get_train(dataroot=DATAROOT):
        return CelebaWrapper.CelebaLMBDDataset(
            LMDBDataset(
                root=CelebaWrapper.root(dataroot),
                name="celeba64",
                split="train",
                transform=CelebaWrapper.train_transform,
                is_encoded=True,
            )
        )

    @staticmethod
    def get_test(dataroot=DATAROOT):
        return CelebaWrapper.CelebaLMBDDataset(
            LMDBDataset(
                root=CelebaWrapper.root(dataroot),
                name="celeba64",
                split="test",
                transform=CelebaWrapper.train_transform,
                is_encoded=True,
            )
        )


def num_samples(dataset, split):
    """Returns the number of samples in the dataset"""
    if dataset == "celeba":
        # return 27000 if train else 3000
        pass
    elif dataset == "celeba64":
        if split == "train":
            return 162770
        elif split == "validation":
            return 19867
        elif split == "test":
            return 19962
    else:
        raise NotImplementedError(f"dataset {dataset} is unknown")


class LMDBDataset(data.Dataset):
    """Dataset for LMDB files"""
    def __init__(self, root, name="", split="train", transform=None, is_encoded=False):
        self.name = name
        self.split = split
        self.transform = transform
        if self.split in ["train", "validation", "test"]:
            lmdb_path = path.join(root, f"{self.split}.lmdb")
        else:
            print("invalid split param")
        self.data_lmdb = lmdb.open(
            lmdb_path,
            readonly=True,
            max_readers=1,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.is_encoded = is_encoded

    def __getitem__(self, index):
        target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert("RGB")
            else:
                img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return num_samples(self.name, self.split)
