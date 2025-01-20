""" Class to extract OOD scores from a model."""
import random
from os import makedirs, path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ood_scores.scores.g_nll import OODScores
from utilities.routes import PathCreator


class OODScoresExtractor(OODScores):
    """ Class to extract OOD scores from a model."""

    def __init__(self, model, device):
        super().__init__(model, device)
        self.model = self.model.to(self.device).eval()
        self.path_creator = PathCreator()
        self.fit_ds_name = None

    def ood_scores_on_batches(self, dataset, batch_size, means, variances, num_samples, checkpoint, fit=False):
        """Compute OOD scores for each sample in the batch"""

        checkpoint_number = int(''.join(filter(str.isdigit, checkpoint)))
        ds_name = getattr(dataset, 'name', dataset.__class__.__name__).lower()
        # Compute OOD scores for Fit dataset
        if fit:
            self.fit_ds_name = ds_name
            output_dir = self.path_creator.model_dataset_path(self.model.__class__.__name__.lower(), self.fit_ds_name)
            output_dir = path.join(output_dir, str(batch_size))
            if not path.exists(output_dir):
                makedirs(output_dir)
            file_dir = f"{output_dir}/ood_scores_fit_samples_b{batch_size}_{self.fit_ds_name}_using_checkpoint_{checkpoint_number}.pth"
            if not path.exists(file_dir):
                ood_scores_fit_samples, features, num_features, features_scalar  = self.run_ood_scores(dataset, batch_size, means, variances, num_samples)
                torch.save({
                    'ood_scores': ood_scores_fit_samples,
                    'features': features,
                    'num_features': num_features,
                    'features_scalar': features_scalar
                }, file_dir)
                # torch.save(ood_scores_fit_samples, file_dir)
                print(f"fit_ood_scores of {self.fit_ds_name} is saved in\n {file_dir}")
            else:
                print(f"fit_ood_scores of {self.fit_ds_name} already exists in\n {file_dir}")
        else:
            # Compute OOD scores for Test dataset
            output_dir = self.path_creator.model_dataset_path(self.model.__class__.__name__.lower(), self.fit_ds_name)
            output_dir = path.join(output_dir, str(batch_size))
            if not path.exists(output_dir):
                makedirs(output_dir)
            file_dir = f"{output_dir}/ood_scores_test_samples_b{batch_size}_{ds_name}_on_{self.fit_ds_name}_using_checkpoint_{checkpoint_number}.pth"
            if not path.exists(file_dir):
                ood_scores_test_samples, features, num_features, features_scalar = self.run_ood_scores(dataset, batch_size, means, variances, num_samples)
                torch.save({
                    'ood_scores': ood_scores_test_samples,
                    'features': features,
                    'num_features': num_features,
                    'features_scalar': features_scalar
                }, file_dir)
                # torch.save(ood_scores_test_samples, file_dir)
                print(f"Test_ood_scores of {ds_name} is saved in\n {file_dir}")
            else:
                print(f"Test_ood_scores of {ds_name} already exists in\n {file_dir}")

    def run_ood_scores(self, ds, b_size, means, variances, num_samples):
        """Compute OOD scores for each sample in the batch"""
        # Create a subset of the test_dataset using the random indices
        random_indices = random.sample(range(len(ds)), num_samples)
        subset = Subset(ds, random_indices)
        loader = DataLoader(subset, batch_size=b_size, shuffle=False)
        ds_name = getattr(ds, 'name', ds.__class__.__name__).lower()

        ood_scores_test_samples = []
        for samples, _ in tqdm(loader, desc=f"Calculating ood_scores on {ds_name} samples"):
            samples = samples.to(self.device)
            # Compute the OOD scores for the test samples
            ood_scores, features, num_features, features_scalar = self.ood_score(samples, means, variances)
            ood_scores_test_samples.append(ood_scores)
        return ood_scores_test_samples, features, num_features, features_scalar
