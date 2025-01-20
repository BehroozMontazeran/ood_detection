"""Utility functions for data loading and manipulation."""

import warnings

import torch

from data.datasets import (CelebAWrapper, CIFAR10Wrapper, FashionMNISTWrapper,
                           FlippedOmniglotWrapper, GTSRBWrapper,
                           ImageNet32Wrapper, MNISTWrapper, OmniglotWrapper,
                           SVHNWrapper)#, MixedWrapper)
                        #  get_celeba, get_cifar10,
                        #    get_fashionmnist, get_flipped_omniglot,
                        #    get_imagenet32, get_mnist, get_omniglot, get_svhn)
from utilities.routes import PROJECT_ROOT

# Dict that maps name string -> DataSetWrapper for the given datasets.

to_dataset_wrapper = {
    DS_Wrapper.name: DS_Wrapper for DS_Wrapper in
    [
        # MixedWrapper,
        ImageNet32Wrapper,
        SVHNWrapper,
        CelebAWrapper,
        CIFAR10Wrapper,
        GTSRBWrapper,
        # MNISTWrapper,
        # FashionMNISTWrapper,
        # OmniglotWrapper,
        # FlippedOmniglotWrapper
    ]
}


dataset_names = set(sorted(to_dataset_wrapper.keys()))


def get_image_shape(dataset_name):
    return to_dataset_wrapper[dataset_name].image_shape


def get_test_dataset(dataset_name):
    return to_dataset_wrapper[dataset_name].get_test()


def get_dataset(dataset_name, split="test"):
    wrapper = to_dataset_wrapper[dataset_name]
    if split == "test":
        return wrapper.get_test()
    elif split == "train":
        return wrapper.get_train()
    else:
        raise ValueError(f"requested dataset split '{split}' was not recognised.")


svhn_path = "SVHN"
cifar_path = "CIFAR10"


def get_vanilla_dataset(dataset_name, return_test=True, dataroot=PROJECT_ROOT):
    warnings.warn("DEPRECATED. Use  instead.", DeprecationWarning)

    dataset_getter = {
        "cifar": get_cifar10,
        "svhn": get_svhn,
        "celeba": get_celeba,
        "imagenet32": get_imagenet32,
        "FashionMNIST": get_fashionmnist,
        "MNIST": get_mnist,
        "Omniglot": get_omniglot,
        "flipped_Omniglot": get_flipped_omniglot,
    }[dataset_name]

    if dataset_name in ["cifar", "svhn"]:
        _, _, train, test = dataset_getter(False, dataroot, True)
    else:
        _, _, train, test = dataset_getter(dataroot)

    if return_test:
        return test
    else:
        return train


# TODO: depricated functions here


def vanilla_test_cifar(dataroot="../"):
    _, _, _, test_cifar = get_cifar10(False, dataroot, True)
    return test_cifar


def vanilla_test_svhn(dataroot="../"):
    _, _, _, test_svhn = get_svhn(False, dataroot, True)
    return test_svhn


def vanilla_test_celeba(dataroot="../"):
    _, _, _, test_celeba = get_celeba(dataroot)
    return test_celeba


def vanilla_test_imagenet32(dataroot="../"):
    _, _, _, test_imagenet32 = get_imagenet32(dataroot)
    return test_imagenet32


def vanilla_test_FashionMNIST(dataroot="../"):
    _, _, _, test_FashionMNIST = get_fashionmnist(dataroot)
    return test_FashionMNIST


def vanilla_test_MNIST(dataroot="../"):
    _, _, _, test_MNIST = get_mnist(dataroot)
    return test_MNIST


def vanilla_test_Omniglot(dataroot="../"):
    _, _, _, test_Omniglot = get_omniglot(dataroot)
    return test_Omniglot


def vanilla_test_flipped_Omniglot(dataroot="../"):
    _, _, _, test_flipped_Omniglot = get_flipped_omniglot(dataroot)
    return test_flipped_Omniglot


def get_vanilla_dataset_depreciated(dataset_name, dataroot="../"):
    warnings.warn("DEPRECATED. Use Imagenet32_Wrapper.get_all instead.", DeprecationWarning)
    return {
        "cifar": vanilla_test_cifar,
        "svhn": vanilla_test_svhn,
        "celeba": vanilla_test_celeba,
        "imagenet32": vanilla_test_imagenet32,
        "FashionMNIST": vanilla_test_FashionMNIST,
        "MNIST": vanilla_test_MNIST,
        "Omniglot": vanilla_test_Omniglot,
        "flipped_Omniglot": vanilla_test_flipped_Omniglot,
    }[dataset_name](dataroot=dataroot)


class SampleDataset:
    """A dataset that generates samples from a model"""
    def __init__(self, model, batch_count=128, temp=1):
        """batch_count is the number of 32-length batches to generate"""
        super().__init__()
        self.batch_count = batch_count
        self.samples = []

        for _ in range(self.batch_count):
            imgs = model.generate_sample(32).cpu()

            for img in imgs:
                self.samples.append(img)

    def __len__(self):
        return self.batch_count * 32

    def __getitem__(self, item):
        return self.samples[item], torch.zeros(10)


class RandomNoiseDataset:
    """A dataset that generates random noise samples"""
    def __init__(self, image_shape):
        super().__init__()
        self.image_shape = image_shape
        print(f"image_shape: {self.image_shape}")

    def __len__(self):
        return 512

    def __getitem__(self, item):
        means = torch.zeros(self.image_shape)
        stds = torch.ones(self.image_shape) / 5
        return torch.normal(means, stds), torch.zeros(10)
