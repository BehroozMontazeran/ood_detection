
import re

from os import path
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

OUTPUT_DIR = path.join(PROJECT_ROOT, "output")
DATAROOT = path.join(PROJECT_ROOT, "data")
MODELS_DIR = path.join(PROJECT_ROOT, "models")

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

# GRADIENTS_DIR = "./anomaly_methods/gradients/serialised_gradients/"

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
        else:
            return None
