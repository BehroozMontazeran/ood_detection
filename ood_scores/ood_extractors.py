""" Class to extract OOD scores from a model."""
import random

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from ood_scores.scores.g_nll import OodScores
from utils.routs import OUTPUT_DIR, PathCreator


class OodScoresExtractor(OodScores):
    """ Class to extract OOD scores from a model."""

    def __init__(self, model, device):
        super().__init__(model, device)
        self.model = self.model.to(self.device).eval()

    def ood_scores_on_batches(self, fit_dataset, test_dataset, batch_size, miu, var):
        """Compute OOD scores for each sample in the batch"""

        fit_ds = fit_dataset
        test_ds = test_dataset
        b_size = batch_size
        means = miu
        variances = var

        min_samples = min(len(fit_ds), len(test_ds))

        # Compute OOD scores for Fit dataset
        ood_scores_fit_samples, fit_ds_name = self.run_ood_scores(fit_ds, b_size, means, variances, min_samples)
        # Save ood_scores_fit_samples
        split = PathCreator().dataset_details(fit_ds_name)
        file_name = f"{OUTPUT_DIR}/ood_scores_fit_samples_{b_size}_{fit_ds_name}_from_{split}_dataset.pth"
        torch.save(ood_scores_fit_samples, file_name)
        print(f"Saved {file_name}")

        # Compute OOD scores for Test dataset
        ood_scores_test_samples, test_ds_name = self.run_ood_scores(test_ds, b_size, means, variances, min_samples)
        # Save ood_scores_test_samples
        file_name = f"{OUTPUT_DIR}/ood_scores_test_samples_{b_size}_{test_ds_name}_on_{fit_ds_name}.pth"
        torch.save(ood_scores_test_samples, file_name)
        print(f"Saved {file_name}")



    def run_ood_scores(self, ds, b_size, means, variances, num_samples):
        """Compute OOD scores for each sample in the batch"""
        # Create a subset of the test_dataset using the random indices
        random_indices = random.sample(range(len(ds)), num_samples)
        subset = Subset(ds, random_indices)
        loader = DataLoader(subset, batch_size=b_size, shuffle=False)
        ds_name = loader.dataset.__class__.__name__
        # test_loader = DataLoader(test_svhn, batch_size=b_size, shuffle=False)

        ood_scores_test_samples = []
        for samples, _ in tqdm(loader, desc=f"Calculating ood_scores on {ds_name} samples"):
            samples = samples.to(self.device)
            # Compute the OOD scores for the test samples
            ood_scores_test_samples.append(self.ood_score(samples, means, variances))
        return ood_scores_test_samples, ds_name


        # # Define DataLoader for the test set
        # random_indices = random.sample(range(len(test_cifar)), num_samples)
        # test_subset = Subset(test_cifar, random_indices)
        # fit_loader = DataLoader(test_subset, batch_size=b_size, shuffle=False)

        # # fit_loader = DataLoader(test_cifar, batch_size=batch_size, shuffle=False)

        # ood_scores_fit_samples = []
        # for fit_samples, _ in tqdm(fit_loader, desc="Calculating ood_scores on fit samples"):
        #     fit_samples = fit_samples.to(device)
        #     # Compute the OOD scores for the fit samples
        #     ood_scores_fit_samples.append(ood_score(model, fit_samples, means, variances, device))
        # # Save ood_scores_fit_samples
        # torch.save(ood_scores_fit_samples, 'output/ood_scores_fit_samples_1_cifar10_from_test_dataset.pth')
        # print("Saved ood_scores_fit_samples_1_cifar10_from_test_dataset.pth")

        # # print(f"{ood_scores_test_samples=}")
        # # print(f"{ood_scores_fit_samples=}")
