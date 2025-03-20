"""Utility functions for data loading and manipulation."""

import warnings

import torch

from data.datasets import (CelebAWrapper, CIFAR10Wrapper, FashionMNISTWrapper,
                           FlippedOmniglotWrapper, GTSRBWrapper,
                           ImageNet32Wrapper, MNISTWrapper, OmniglotWrapper,
                           SVHNWrapper, KMNISTWrapper)
from utilities.routes import PROJECT_ROOT

# Dict that maps name string -> DataSetWrapper for the given datasets.

to_dataset_wrapper = {
    DS_Wrapper.name: DS_Wrapper for DS_Wrapper in
    [
        ImageNet32Wrapper,
        SVHNWrapper,
        CelebAWrapper,
        CIFAR10Wrapper,
        GTSRBWrapper,
    ]
}
dataset_names_ch3 = set(sorted(to_dataset_wrapper.keys()))

to_dataset_wrapper = {
    DS_Wrapper.name: DS_Wrapper for DS_Wrapper in
    [
        MNISTWrapper,
        FashionMNISTWrapper,
        OmniglotWrapper,
        KMNISTWrapper,
    ]
}

dataset_names_ch1 = set(sorted(to_dataset_wrapper.keys()))


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
