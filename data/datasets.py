"""
Defines the Datasets used in the model training and evaluation.
The idea comes from
https://github.com/SamD770/Generative-Models-Knowledge
"""

import os
import pickle
import random
import shutil
import zipfile
from abc import ABC, abstractmethod
from io import BytesIO
from os import listdir, path, makedirs

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms import functional

from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from torchvision.datasets import DatasetFolder

from torch.utils.data import TensorDataset
from tqdm import tqdm


from utilities.routes import DATAROOT

# All greyscale datasets are scaled from [0, 255] to [0, 1]
# All color datasets are scaled from [0, 255] to [-0.5, 0.5]


N_BITS = 8


def ppm_loader(filepath):
    """Loads a .ppm image file."""
    return Image.open(filepath).convert("RGB")


def preprocess(x):
    """Preprocessing used in the Glow code
    based on:
    https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78
"""
    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2**N_BITS
    if N_BITS < 8:
        x = torch.floor(x / 2 ** (8 - N_BITS))
    x = x / n_bins - 0.5

    return x

def postprocess(x):
    """Postprocessing used in the Glow code"""
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
    def get_val(cls, dataroot: str = DATAROOT, transform=None, download=True) -> tuple:
        """
        Returns a tuple of dataset info: (image shape, num classes, train dataset, test dataset)
        """
        return (cls.image_shape, cls.num_classes,
                None,
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

class OmniglotWrapper(DatasetWrapper):
    """ Omniglot dataset wrapper """
    name = "omniglot"
    image_shape = (28, 28, 1)
    num_classes = 10

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

class FlippedOmniglotWrapper(OmniglotWrapper):
    """ Flipped Omniglot dataset wrapper """
    name = "flipped_omniglot"

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
    
    
class CIFAR10DatasetFolder(DatasetFolder):
    """Custom DatasetFolder for CIFAR10 with a specific name."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "cifar10"

    def __repr__(self):
        return f"Dataset({self.name})"


# Normal way of loading train and test datasets
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
            # apply_perturbation, # When no perturbation is needed, remove this part
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
        return [transforms.RandomAffine(0, translate=(0.1, 0.1)),]#,apply_perturbation,]

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

class GTSRBDatasetFolder(DatasetFolder):
    """Custom DatasetFolder for GTSRB with a specific name."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "gtsrb"  # Define the name of the dataset

    def __repr__(self):
        return f"Dataset({self.name})"


class GTSRBWrapper(DatasetWrapper):
    """For the German Traffic Sign Recognition Benchmark (Houben et al. 2013)"""
    name = "gtsrb"
    image_shape = (32, 32, 3)
    num_classes = 40
    pixel_range = (-0.5, 0.5)
    perturb_rate = 0.1

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
            # apply_perturbation,
        ]

    @staticmethod
    def default_preprocessing() -> list:
        return [transforms.ToTensor(), transforms.Resize((GTSRBWrapper.image_shape[0], GTSRBWrapper.image_shape[1])), preprocess]

 
    @staticmethod
    def get_train(dataroot: str=DATAROOT, transform=None, augment=True, download: bool=True) -> Dataset:
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


        lmdb_path = path.join(ImageNet32Wrapper.root(dataroot), 'train.lmdb')
        batch_folder = path.join(ImageNet32Wrapper.root(dataroot), 'train_batches')

        # If LMDB doesn't exist, create it
        if not path.exists(lmdb_path):
            ImageNet32Wrapper.create_lmdb_from_batches(lmdb_path, batch_folder, transform)

        # Define transformations
        if transform is None:
            transform = ImageNet32Wrapper.create_transform(augment)

        dataset = ImageNet32Wrapper(lmdb_path, transform)
        images, labels = [], [] 
        for i in tqdm(range(len(dataset)), desc="Loading training data from ImageNet32"):
            img, label = dataset[i]
            images.append(img)
            labels.append(label)
        tensor_dataset=torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))
        tensor_dataset.name = ImageNet32Wrapper.name  # Set the name attribute directly
        return tensor_dataset

    @staticmethod
    def get_test(dataroot: str = DATAROOT, transform=None, download: bool = False) -> Dataset:
        """Returns the validation dataset."""

        val_zip_path = path.join(ImageNet32Wrapper.root(dataroot), 'Imagenet32_val.zip')
        extract_path = path.join(ImageNet32Wrapper.root(dataroot), 'val_batches')

        # Extract the validation data if it hasn't been extracted yet
        if not path.exists(extract_path):
            ImageNet32Wrapper.extract_zip(val_zip_path, extract_path)


        lmdb_path = path.join(ImageNet32Wrapper.root(dataroot), 'val.lmdb')
        batch_folder = path.join(ImageNet32Wrapper.root(dataroot), 'val_batches')

        # If LMDB doesn't exist, create it
        if not path.exists(lmdb_path):
            ImageNet32Wrapper.create_lmdb_from_batches(lmdb_path, batch_folder, transform)

        # Define transformations
        if transform is None:
            transform = ImageNet32Wrapper.create_transform(augment=False)

        dataset = ImageNet32Wrapper(lmdb_path, transform)
        images, labels = [], []
        for i in tqdm(range(len(dataset)), desc="Loading validation data of ImageNet32"):
            img, label = dataset[i]
            images.append(img)
            labels.append(label)

        tensor_dataset=torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))
        tensor_dataset.name = ImageNet32Wrapper.name  # Set the name attribute directly
        return tensor_dataset


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
        self.transform = transform
        self.lmdb_path = lmdb_path
        # Open LMDB after ensuring it exists
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = txn.stat()["entries"]
            print(f"Total entries in LMDB split): {self.length}")

    @staticmethod
    def root(dataroot):
        """Returns the root directory for CelebA."""
        return path.join(dataroot, "CELEBA")

    @staticmethod
    def get_split_files(img_folder: str, attributes: dict, split_ratio: float=0.8):
        """Loads or creates train and test split files and physically splits images."""
        train_split_path = path.join(img_folder, "train_imgs")
        test_split_path = path.join(img_folder, "test_imgs")

        if path.exists(train_split_path) and path.exists(test_split_path):
            # If folders already exist, get list of files from them
            print("Loading existing train/test split.")
            train_files = [path.join("train_imgs", fname) for fname in os.listdir(train_split_path)]
            test_files = [path.join("test_imgs", fname) for fname in os.listdir(test_split_path)]
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
                source_path = path.join(img_folder, img_name)
                target_path = path.join(train_split_path, img_name)
                shutil.move(source_path, target_path)  # Move file to train folder

            for img_name in tqdm(test_files, desc="Moving test files"):
                source_path = path.join(img_folder, img_name)
                target_path = path.join(test_split_path, img_name)
                shutil.move(source_path, target_path)  # Move file to test folder

            # Update paths with new locations
            train_files = [path.join("train_imgs", fname) for fname in train_files]
            test_files = [path.join("test_imgs", fname) for fname in test_files]

        return train_files, test_files

    @staticmethod
    def default_preprocessing() -> list:
        return [CropCelebA64(),transforms.Resize((CelebAWrapper.image_shape[0], CelebAWrapper.image_shape[1])),
                 transforms.ToTensor(), preprocess]

    @staticmethod
    def extract_zip(zip_path: str, extract_to: str) -> None:
        """Extracts a zip file to a specified directory."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    
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
                img_path = path.join(images_folder, img_name)
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
        # return len(self.data_files)
        return self.length

    @staticmethod
    def get_train(dataroot: str = DATAROOT, transform=None, augment: bool = True, download: bool = False) -> Dataset:
        """Returns the training dataset."""
        train_zip_path = path.join(CelebAWrapper.root(dataroot), 'archive.zip')
        images_folder = path.join(CelebAWrapper.root(dataroot), 'img')
        attr_file = path.join(CelebAWrapper.root(dataroot), "list_attr_celeba.csv")

        if not path.exists(images_folder) or not path.exists(attr_file):
            CelebAWrapper.extract_zip(train_zip_path, images_folder)

        lmdb_path = path.join(CelebAWrapper.root(dataroot), 'train.lmdb')
        if not path.exists(lmdb_path):
            attributes = CelebAWrapper.load_attributes(attr_file)
            train_files, _ = CelebAWrapper.get_split_files(images_folder, attributes)
            CelebAWrapper.create_lmdb_from_images(lmdb_path, images_folder, train_files , attributes, transform)

        if transform is None:
            transform = CelebAWrapper.create_transform(augment)

        dataset = CelebAWrapper(lmdb_path, transform)
        images, labels = [], [] 
        for i in tqdm(range(len(dataset)), desc="Loading training data of CelebA"):
            img, label = dataset[i]
            images.append(img)
            labels.append(label)
        tensor_dataset=torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))
        tensor_dataset.name = CelebAWrapper.name  # Set the name attribute directly
        return tensor_dataset

    @staticmethod
    def get_test(dataroot: str=DATAROOT, transform=None, download: bool = False) -> Dataset:
        """Returns the validation dataset."""
        val_zip_path = path.join(CelebAWrapper.root(dataroot), 'archive.zip')
        images_folder = path.join(CelebAWrapper.root(dataroot), 'img')
        attr_file = path.join(CelebAWrapper.root(dataroot), "list_attr_celeba.csv")

        if not path.exists(images_folder) or not path.exists(attr_file):
            CelebAWrapper.extract_zip(val_zip_path, images_folder)

        lmdb_path = path.join(CelebAWrapper.root(dataroot), 'val.lmdb')
        if not path.exists(lmdb_path):
            attributes = CelebAWrapper.load_attributes(attr_file)
            _, test_files = CelebAWrapper.get_split_files(images_folder, attributes)
            CelebAWrapper.create_lmdb_from_images(lmdb_path, images_folder, test_files, attributes, transform)

        if transform is None:
            transform = CelebAWrapper.create_transform(augment=False)

        dataset = CelebAWrapper(lmdb_path, transform)
        images, labels = [], [] 
        for i in tqdm(range(len(dataset)), desc="Loading validation data of CelebA"):
            img, label = dataset[i]
            images.append(img)
            labels.append(label)
        tensor_dataset=torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))
        tensor_dataset.name = CelebAWrapper.name  # Set the name attribute directly
        return tensor_dataset

class MixedWrapper(DatasetWrapper):
    """Mixed dataset wrapper."""

    name = "mixed"
    image_shape = (32, 32, 3)
    num_classes = 1000

    def __init__(self, dataset1, dataset2, balance_ratio=0.5):
        """
        Initializes the combined dataset.

        Args:
            dataset1: First dataset (ImageNet32).
            dataset2: Second dataset (GTSRB).
            balance_ratio: Fraction of samples from dataset1 in each batch.
                            A value of 0.5 means equal samples from both datasets.
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.balance_ratio = balance_ratio
        self.length1 = len(dataset1)
        self.length2 = len(dataset2)
        self.total_length = self.length1 + self.length2

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        # Decide which dataset to sample from based on the balance ratio
        if torch.rand(1).item() < self.balance_ratio:
            dataset, idx = self.dataset1, index % self.length1
        else:
            dataset, idx = self.dataset2, index % self.length2
        image, label = dataset[idx]
        
        # Ensure the label is a tensor
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        
        return image, label

    @staticmethod
    def get_train(dataroot: str=DATAROOT, transform=None, augment: bool=True, download: bool=True) -> Dataset:
        """Loads and combines ImageNet32 and GTSRB training datasets."""
        imagenet_train = ImageNet32Wrapper.get_train(dataroot)
        gtsrb_train = GTSRBWrapper.get_train(dataroot)
        
        # Combine with a 50% balance ratio for each dataset
        combined_train_dataset = MixedWrapper(imagenet_train, gtsrb_train, balance_ratio=0.5)
        return combined_train_dataset

    @staticmethod
    def get_test(dataroot: str=DATAROOT, transform=None, download: bool=True) -> Dataset:
        """Loads and combines ImageNet32 and GTSRB test datasets."""
        imagenet_test = ImageNet32Wrapper.get_test(dataroot)
        gtsrb_test = GTSRBWrapper.get_test(dataroot)
        
        # Combine with a 50% balance ratio for each dataset
        combined_test_dataset = MixedWrapper(imagenet_test, gtsrb_test, balance_ratio=0.5)
        return combined_test_dataset



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
