""" Run the main program. """

import argparse
from os import listdir, path

import torch
from torch.utils.data import DataLoader

from models.load_models import LoadModels
from ood_scores.features.feature_extractors import FeatureExtractor
from ood_scores.ood_extractors import OODScoresExtractor
from utilities.plot import process_and_plot
from utilities.routes import OUTPUT_DIR
from utilities.utils import dataset_names_ch3, dataset_names_ch1


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
        "--num_channels",
        type=int,
        default=3,
        choices=[1, 3],
        help="Number of channels in the input images."
    )
    parser.add_argument(
        "--fit_dataset",
        type=str,
        help="Dataset to use for fitting for 3channels from [cifar10, celeba, imagenet32, gstrb, svhn] and 1channels from [FashionMNIST, MNIST, Omniglot, KMNIST]"
    )
    parser.add_argument(
        "--ood_batch_size",
        type=int,
        default=1,
        help="Batch size for OOD score computation."
    )
    parser.add_argument(
        "--ood_num_samples",
        type=int,
        default=1000,
        help="Number of samples for OOD score computation."
    )
    parser.add_argument(
        "--plot",
        type=bool,
        default=True,
        help="Whether to plot the results."
    )
    parser.add_argument(
        "--loop_over_all",
        type=bool,
        default=False,
        help="Whether to loop over all datasets[cifar10, celeba, imagenet32, gstrb, svhn], ood_batch_sizes {1,5} and checkpoints {0, -1}."
    )
    args = parser.parse_args()

    # Select dataset list based on num_channels
    dataset_names = dataset_names_ch3 if args.num_channels == 3 else dataset_names_ch1

    # Validate fit_dataset selection
    if args.fit_dataset not in dataset_names:
        raise ValueError(f"Invalid fit_dataset '{args.fit_dataset}'. Choose from: {dataset_names}")

    loop_over_all = args.loop_over_all

    if loop_over_all:
        fit_dataset_names = dataset_names
        for fit_dataset_name in fit_dataset_names:
            data_path = path.join(OUTPUT_DIR, f"{args.model_type}_{fit_dataset_name}")
            checkpoints = [f for f in listdir(data_path) if f.endswith('.pt')]
            for ood_batch_size in [1, 5]:
                calculate_results(args, fit_dataset_name=fit_dataset_name, data_path=data_path, ood_batch_size=ood_batch_size, checkpoints=checkpoints)
    else:
        calculate_results(args)


def calculate_results(args, fit_dataset_name=None, data_path=None, ood_batch_size=None, checkpoints=None):
    """ Calculate the OOD scores for the given arguments. """
    load_models = LoadModels()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ood_batch_size = int(args.ood_batch_size) if ood_batch_size is None else int(ood_batch_size)
    num_samples = int(args.ood_num_samples)
    fit_dataset_name = args.fit_dataset if fit_dataset_name is None else fit_dataset_name
    model_type = args.model_type
    plot = args.plot
    # List the name of checkpoints for each subfolder
    data_path = path.join(OUTPUT_DIR, f"{model_type}_{fit_dataset_name}") if data_path is None else data_path
    checkpoints = [f for f in listdir(data_path) if f.endswith('.pt')] if checkpoints is None else checkpoints

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
        means, variances, _ = feature_extractor.fit_gaussians_to_log_features(fit_loader, data_path, ood_batch_size, checkpoint)
        ood_scores_extractor.ood_scores_on_batches(fit_test, ood_batch_size, means, variances, num_samples=num_samples, checkpoint=checkpoint, fit=True)

        # Load other dataset_name as test
        dataset_names.remove(fit_dataset_name)
        test_dataset_names = dataset_names
        # Process OOD scores for each test dataset
        for test_dataset_name in test_dataset_names:
            test_ds = load_models.load_test_dataset(model_type, test_dataset_name)
            ood_scores_extractor.ood_scores_on_batches(test_ds, ood_batch_size, means, variances, num_samples=num_samples, checkpoint=checkpoint, fit=False)
        dataset_names.add(fit_dataset_name)


    if plot:
        # Plot Histograms of OOD scores and ROC curves
        d_path = path.join(data_path, str(ood_batch_size))
        process_and_plot(d_path)

if __name__ == '__main__':
    main()
