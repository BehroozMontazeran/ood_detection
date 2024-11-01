""" Script to load models from disk. """

import json
from os import path

import torch

from models.glow_model.model import Glow
from models.glow_model.train import get_ds_params
from utilities.routes import PathCreator


class LoadModels:
    """ Class to load models from disk. """
    def __init__(self):
        self.path_creator = PathCreator()
        self.transform = None

    def glow(self, checkpoint="glow_checkpoint_195250.pt", fit_dataset_name = 'cifar10'):
        """ Load the glow model from disk based on fit and test dataset names. """

        image_shape, num_classes, train_ds, test_ds, hparams, output_dir = self.load_fit_dataset("glow", fit_dataset_name)

        model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                    hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                    hparams['learn_top'], hparams['y_condition'])
        chekpoint_dir = path.join(output_dir, checkpoint)
        model.load_state_dict(torch.load(chekpoint_dir)['model']) # Load only model part
        model.set_actnorm_init()

        return  model, train_ds, test_ds

    def load_fit_dataset(self, model_type,  dataset_name):
        """ Load the dataset from disk. """
        output_dir = self.path_creator.model_dataset_path(model_type, dataset_name)

        with open(path.join(output_dir, "hparams.json"), 'r', encoding='utf-8') as json_file:
            hparams = json.load(json_file)

        image_shape, num_classes, train_ds, test_ds = get_ds_params(dataset_name, hparams['dataroot'], self.transform, hparams['augment'], hparams['download'], mode='test')

        return image_shape, num_classes, train_ds, test_ds, hparams, output_dir
    
    def load_test_dataset(self, model_type,  dataset_name):
        """ Load the dataset from disk. """
        output_dir = self.path_creator.model_dataset_path(model_type, dataset_name)

        with open(path.join(output_dir, "hparams.json"), 'r', encoding='utf-8') as json_file:
            hparams = json.load(json_file)

        _, _, _, test_ds = get_ds_params(dataset_name, hparams['dataroot'], self.transform, hparams['augment'], hparams['download'], mode='test')

        return test_ds
