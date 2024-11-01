""" Feature Extractor Base Class """
import torch
from tqdm import tqdm


class FeatureExtractor:
    """ Base class for feature extractors """
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(self.device).eval()


    def gradient_features(self, inputs):
        """ Extracts the squared L2 norms of gradients of the model's output w.r.t. the input """
        # Set the model to evaluation mode
        inputs = inputs.to(self.device)

        # Reset gradients
        self.model.zero_grad()
        # Enable gradient tracking
        inputs.requires_grad = True

        # Forward pass through the model
        outputs = self.model(inputs)
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
        for param in self.model.parameters():
            if param.grad is not None:
                # Layer-wise L2 norm of gradients
                squared_layer_norm = torch.norm(param.grad)**2
                features.append(squared_layer_norm)  # Store the layer-wise norm
                features_scalar.append(squared_layer_norm.item())  # Convert to a scalar and store
                num_features.append(param.grad.numel())  # Store the number of features

        return features, num_features, features_scalar


    def fit_gaussians_to_log_features(self, fit_loader):
        """ Fits Gaussians to the log of the gradient features """
        # Initialize a list to hold log features for each layer
        # Use a list of tensors initialized with empty tensors to accumulate log features
        all_features = None
        fit_ds_name = getattr(fit_loader.dataset, 'name', fit_loader.dataset.__class__.__name__).lower()
        # Loop through batches in the fit dataset
        for inputs, _ in tqdm(fit_loader, desc=f'Fitting Gaussians to log features {fit_ds_name}'):
            inputs = inputs.to(self.device)

            # Get the gradient features for the batch
            features, _, _ = self.gradient_features(inputs)

            # Take the log of each feature in the batch
            log_features = [torch.log(f + 1e-10) for f in features]  # Add small value to avoid log(0)

            # If all_features is None, initialize it with the first batch's log features
            if all_features is None:
                all_features = [f.unsqueeze(0) for f in log_features]  # Start a new list of tensors
            else:
                # Concatenate the new log features with existing ones for each layer
                for i, feature in enumerate(all_features):
                    all_features[i] = torch.cat((feature, log_features[i].unsqueeze(0)), dim=0)

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
