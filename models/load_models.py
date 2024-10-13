""" Script to load models from disk. """

import json
import torch

from models.glow_model.datasets import get_CIFAR10, get_SVHN
from models.glow_model.model import Glow
from utils.routs import OUTPUT_DIR


class LoadModels:
    """ Class to load models from disk. """
    def __init__(self):
        pass

    def glow(self, checkpoint="glow_checkpoint_195250.pt", fit_dataset_name = 'cifar10', test_dataset_name = 'svhn'):
        """ Load the glow model from disk based on fit and test dataset names. """

        fit_train_ds = None
        fit_test_ds = None
        test_train_ds = None
        test_test_ds = None

        with open(OUTPUT_DIR + f'{fit_dataset_name}_hparams.json') as json_file:  
            hparams = json.load(json_file)

        if fit_dataset_name == 'cifar10':
            image_shape, num_classes, fit_train_ds, fit_test_ds = get_CIFAR10(hparams['augment'], hparams['dataroot'], hparams['download'])
            if test_dataset_name == 'svhn':
                image_shape, num_classes, test_train_ds, test_test_ds = get_SVHN(hparams['augment'], hparams['dataroot'], hparams['download'])
        elif fit_dataset_name == 'svhn':
            image_shape, num_classes, fit_train_ds, fit_test_ds = get_SVHN(hparams['augment'], hparams['dataroot'], hparams['download'])
            if test_dataset_name == 'cifar10':
                image_shape, num_classes, test_train_ds, test_test_ds = get_CIFAR10(hparams['augment'], hparams['dataroot'], hparams['download'])
        else:
            raise ValueError(f"Dataset name {fit_dataset_name} or {test_dataset_name} not recognized.")

        model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                    hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                    hparams['learn_top'], hparams['y_condition'])

        model.load_state_dict(torch.load(OUTPUT_DIR + checkpoint)['model']) # Load only model part
        model.set_actnorm_init()

        return  model, fit_train_ds, fit_test_ds, test_train_ds, test_test_ds
