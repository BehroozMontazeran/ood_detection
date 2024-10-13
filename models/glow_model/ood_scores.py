import torch
from tqdm import tqdm

import json
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from datasets import get_CIFAR10, get_SVHN, postprocess
from model import Glow
from torch.utils.data import DataLoader, Subset
import random

def choose_model(ds_name = 'cifar10'):
    device = torch.device("cuda")

    output_folder = 'output/'
    # model_name = 'glow_model_250.pth'
    ds_name = ds_name  # svhn

    if ds_name == 'cifar10':
        model_name = 'glow_checkpoint_195250.pt'

        with open(output_folder + 'cifar10_hparams.json') as json_file:  
            hparams = json.load(json_file)

    elif ds_name == 'svhn':
        model_name = 'glow_checkpoint_286000.pt'

        with open(output_folder + 'svhn_hparams.json') as json_file:  
            hparams = json.load(json_file)
        
    image_shape, num_classes, _, test_cifar = get_CIFAR10(hparams['augment'], hparams['dataroot'], hparams['download'])
    image_shape, num_classes, _, test_svhn = get_SVHN(hparams['augment'], hparams['dataroot'], hparams['download'])


    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                hparams['learn_top'], hparams['y_condition'])

    model.load_state_dict(torch.load(output_folder + model_name)['model']) # Load only model part
    model.set_actnorm_init()

    model = model.to(device)
    model = model.eval()

    return  model, device, test_cifar, test_svhn


def gradient_features(model, inputs, device):
    
    # Set the model to evaluation mode
    model = model.to(device).eval()
    inputs = inputs.to(device)

    # Reset gradients
    model.zero_grad()
    # Enable gradient tracking
    inputs.requires_grad = True

    # Forward pass through the model
    outputs = model(inputs)
    # print(f"{outputs= }")
    
    # If the model outputs a tuple (e.g., logits, auxiliary outputs), extract the logits
    if isinstance(outputs, tuple):
        log_likelihoods = outputs[0]  # Assuming the first element is the relevant output (logits or likelihood)
    else:
        log_likelihoods = outputs  # If it's not a tuple, directly use the output
    # print(f"{log_likelihoods= }")
    # Sum the log-likelihoods to compute the total loss
    loss = torch.sum(log_likelihoods)
    # print(f"{loss= }")
    # Backpropagate to compute gradients
    loss.backward()

    # Calculate layer-wise squared L2 norms of gradients and keep track of the number of features
    features_scalar = []
    features = []
    num_features = []
    for param in model.parameters():
        if param.grad is not None:
            # Layer-wise L2 norm of gradients
            squared_layer_norm = torch.norm(param.grad)**2
            features.append(squared_layer_norm)  # Store the layer-wise norm
            features_scalar.append(squared_layer_norm.item())  # Convert to a scalar and store
            num_features.append(param.grad.numel())  # Store the number of features

    return features, num_features, features_scalar



def fit_gaussians_to_log_features(model, fit_loader, device):
    model.eval()  # Set model to evaluation mode

    # Initialize a list to hold log features for each layer
    # Use a list of tensors initialized with empty tensors to accumulate log features
    all_features = None

    # Loop through batches in the fit dataset
    for inputs, _ in tqdm(fit_loader, desc='Fitting Gaussians to log features'):
        inputs = inputs.to(device)

        # Get the gradient features for the batch
        features, _, _ = gradient_features(model, inputs, device)

        # Take the log of each feature in the batch
        log_features = [torch.log(f + 1e-10) for f in features]  # Add small value to avoid log(0)

        # If all_features is None, initialize it with the first batch's log features
        if all_features is None:
            all_features = [f.unsqueeze(0) for f in log_features]  # Start a new list of tensors
        else:
            # Concatenate the new log features with existing ones for each layer
            for i in range(len(all_features)):
                all_features[i] = torch.cat((all_features[i], log_features[i].unsqueeze(0)), dim=0)

    # Initialize lists to store Gaussian parameters for each layer
    means = []
    variances = []

    # Iterate over layers to calculate mean and variance
    for layer_features in all_features:
        # Compute mean and variance across batches
        layer_mean = torch.mean(layer_features, dim=0)
        layer_variance = torch.var(layer_features, dim=0)

        # Append as tensors to means and variances
        means.append(layer_mean.detach())  # Detach to avoid gradient tracking
        variances.append(layer_variance.detach())

    return means, variances, all_features



def gaussian_negative_log_likelihood(log_features, means, variances):
    ood_scores = []

    # Iterate over each layer's features, means, and variances
    for layer_features, mu, sigma2 in zip(log_features, means, variances):
        # Ensure mu and sigma2 are tensors and have the correct shape
        mu = mu.to(layer_features.device)
        sigma2 = sigma2.to(layer_features.device)

        # Reshape sigma2 to allow broadcasting
        if sigma2.dim() == 1:
            sigma2 = sigma2.view(1, -1)  # Shape (1, num_features)

        # Compute the negative log-likelihood for the layer
        epsilon = 1e-10
        nll = 0.5 * torch.sum(((layer_features - mu) ** 2) / (sigma2+epsilon) + torch.log(2 * torch.pi * (sigma2+epsilon)), dim=0)
        ood_scores.append(nll)
        # print(f"{nll= }")

    # Sum over all layers to get the final OOD score for each sample
    # sfeatures = torch.stack(ood_scores)
    # print(f"{sfeatures.shape},\n {sfeatures= }")
    # print(f"{ood_scores= }")
    
    return torch.sum(torch.stack(ood_scores))

def ood_score(model, new_samples, means, variances, device):
    # model.to(device)
    model.eval()  # Set model to evaluation mode

    # Compute the gradient features for the new batch
    features, _, _ = gradient_features(model, new_samples, device)

    # Take the log of each feature in the batch
    log_features = [torch.log(f + 1e-10) for f in features]  # Add small value to avoid log(0)
    # print(f"{log_features= }")

    # Compute OOD score using Gaussian negative log-likelihood
    ood_scores = gaussian_negative_log_likelihood(log_features, means, variances)

    return ood_scores

def ood_scores_for_batch(md, tst_cifar, tst_svhn, b_size, mns, var, dvc):
    # Compute OOD scores for each sample in the batch
    # TRain on CiFar10 and test on SVHN
    test_cifar = tst_cifar
    test_svhn = tst_svhn
    device = dvc
    model = md.to(device)
    means = mns
    variances = var


    # Randomly select 1000 samples from a dataset
    num_samples = min(len(test_cifar), len(test_svhn))
    batch_size = b_size


    # Create a subset of the fit_dataset using the random indices
    random_indices = random.sample(range(len(test_cifar)), num_samples)
    test_subset = Subset(test_cifar, random_indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # test_loader = DataLoader(test_svhn, batch_size=batch_size, shuffle=False)

    ood_scores_test_samples = []
    for test_samples, _ in tqdm(test_loader, desc="Processing test samples"):
        test_samples = test_samples.to(device)
        # Compute the OOD scores for the test samples
        ood_scores_test_samples.append(ood_score(model, test_samples, means, variances, device))

    # Save ood_scores_fit_samples
    torch.save(ood_scores_test_samples, 'output/ood_scores_test_samples_1_cifar10_on_svhn.pth')
    print("Saved ood_scores_test_samples_1_cifar10_on_svhn.pth")


    # Define DataLoader for the test set
    random_indices = random.sample(range(len(test_svhn)), num_samples)
    test_subset = Subset(test_svhn, random_indices)
    fit_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # fit_loader = DataLoader(test_cifar, batch_size=batch_size, shuffle=False)

    ood_scores_fit_samples = []
    for fit_samples, _ in tqdm(fit_loader, desc="Processing fit samples"):
        fit_samples = fit_samples.to(device)
        # Compute the OOD scores for the fit samples
        ood_scores_fit_samples.append(ood_score(model, fit_samples, means, variances, device))
    # Save ood_scores_fit_samples
    torch.save(ood_scores_fit_samples, 'output/ood_scores_fit_samples_1_svhn_from_test_dataset.pth')
    print("Saved ood_scores_fit_samples_1_svhn_from_test_dataset.pth")



if __name__ == '__main__':

    num_samples = 1000
    batch_size = 1
    ds_name = 'svhn'

    model, device, test_cifar, test_svhn = choose_model(ds_name=ds_name)

    if ds_name == 'cifar10':
        fit_loader = DataLoader(test_cifar, batch_size=batch_size, shuffle=True)
    elif ds_name == 'svhn':
        fit_loader = DataLoader(test_svhn, batch_size=batch_size, shuffle=True)
    # Fit Gaussians to the log of the gradient features
    means, variances, all_features = fit_gaussians_to_log_features(model, fit_loader, device)
    ood_scores_for_batch(model, test_cifar, test_svhn, batch_size, means, variances, device)