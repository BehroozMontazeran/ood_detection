""" Run the main program. """

import argparse
from os import listdir, path

import torch
from torch.utils.data import DataLoader

from models.load_models import LoadModels
from ood_scores.features.feature_extractors import FeatureExtractor
from ood_scores.ood_extractors import OODScoresExtractor
from utilities.routes import OUTPUT_DIR
from utilities.utils import dataset_names, to_dataset_wrapper


def main():
    """ Run the main program. """
    parser = argparse.ArgumentParser()

    # Add an argument for the fitting dataset
    parser.add_argument(
        "--model_type",
        type=str,
        default="glow",
        choices=["glow", "vae", "diffusion"],
        help="Model type to use for fitting. The model type should be one of ['glow', 'vae', 'diffusion']"
    )
    parser.add_argument(
        "--fit_dataset",
        type=str,
        default="cifar10",  # Default fit dataset
        choices=to_dataset_wrapper.keys(),
        help="Dataset to use for fitting."
    )
    parser.add_argument(
        "--ood_batch_size",
        type=str,
        default="1",
        help="Batch size for OOD score computation."
    )
    parser.add_argument(
        "--ood_num_samples",
        type=str,
        default="5000",
        help="Number of samples for OOD score computation."
    )

    args = parser.parse_args()

    load_models = LoadModels()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ood_batch_size = int(args.ood_batch_size)
    num_samples = int(args.ood_num_samples)
    fit_dataset_name = args.fit_dataset
    model_type = args.model_type

    # List the name of checkpoints for each subfolder
    data_path = path.join(OUTPUT_DIR, f"{model_type}_{fit_dataset_name}")
    checkpoints = [f for f in listdir(data_path) if f.endswith('.pt')]

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {data_path}")

    for checkpoint in [checkpoints[0], checkpoints[-1]]:
        print(f"Using checkpoint: {checkpoint}...\n")
        # Load the model with the fitting dataset
        model, _, fit_test = load_models.glow(checkpoint=checkpoint, fit_dataset_name=fit_dataset_name)

        feature_extractor = FeatureExtractor(model, device)
        ood_scores_extractor = OODScoresExtractor(model, device)

        # DataLoader for the fitting dataset
        fit_loader = DataLoader(fit_test, batch_size=ood_batch_size, shuffle=True)

        # Fit Gaussians to the log of the gradient features
        means, variances, _ = feature_extractor.fit_gaussians_to_log_features(fit_loader)
        ood_scores_extractor.ood_scores_on_batches(fit_test, ood_batch_size, means, variances, num_samples=num_samples, checkpoint=checkpoint, fit=True)

        # Load other dataset_name as test
        dataset_names.remove(fit_dataset_name)
        test_dataset_names = dataset_names
        # Process OOD scores for each test dataset
        for test_dataset_name in test_dataset_names:
            test_ds = load_models.load_test_dataset(model_type, test_dataset_name)
            ood_scores_extractor.ood_scores_on_batches(test_ds, ood_batch_size, means, variances, num_samples=num_samples, checkpoint=checkpoint, fit=False)
        dataset_names.add(fit_dataset_name)

if __name__ == '__main__':
    main()
