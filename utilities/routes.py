"""This module contains the paths for the project directories and the class PathCreator"""

import pathlib
import re
from os import path

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()

OUTPUT_DIR = path.join(PROJECT_ROOT, "output")
DATAROOT = path.join(PROJECT_ROOT, "data")
MODELS_DIR = path.join(PROJECT_ROOT, "models")
OUTPUT_IMG = path.join(PROJECT_ROOT, "images")

VAE_ROOT = path.join(MODELS_DIR, "VAE_model")
GLOW_ROOT = path.join(MODELS_DIR, "glow_model")
PIXEL_CNN_ROOT = path.join(MODELS_DIR, "pixelCNN_model")
DIFFUSION_ROOT = path.join(MODELS_DIR, "diffusion_model")

ANOMALY_DIR = path.join(PROJECT_ROOT, "anomaly_methods")

GRADIENTS_DIR = path.join(ANOMALY_DIR, "gradients")
L2_NORMS_DIR = path.join(GRADIENTS_DIR, "L2_norms")
FISHER_NORMS_DIR = path.join(GRADIENTS_DIR, "Fisher_norms")

LIKELIHOODS_DIR = path.join(ANOMALY_DIR, "likelihoods")

SERIALISED_GRADIENTS_DIR = path.join(GRADIENTS_DIR, "serialised_gradients") # Deprecated


PLOTS_DIR = path.join(PROJECT_ROOT, "plots")

class PathCreator:
    """ Base class for creating paths"""
    def __init__(self):
        pass

    def dataset_details(self, ds_str):
        """Extract the dataset details from the dataset string"""
        # Use regular expression to find the split
        match = re.search(r"Split:\s*(\w+)", ds_str)

        if match:
            split = match.group(1)  # Extract the split (either 'Test' or 'Train')
            return split  # Output: Dataset split: Test
        return None

    def model_dataset_path(self, model_name, dataset_name):
        """Create the path for the model dataset"""
        return path.join(OUTPUT_DIR, f"{model_name}_{dataset_name}")
