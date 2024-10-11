

from os import path
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

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
