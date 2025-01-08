"""calculate the OOD score using the Gaussian negative log-likelihood method"""

import torch

from ood_scores.features.feature_extractors import FeatureExtractor


class OODScores(FeatureExtractor):
    """ Base class for ood scores calculators """
    def __init__(self, model, device):
        super().__init__(model, device)
        self.model = self.model.to(self.device).eval()

    def gaussian_negative_log_likelihood(self, log_features, means, variances):
        """ Compute the Gaussian negative log-likelihood for the given log features, means, and variances """
        ood_scores = []

        # Iterate over each layer's features, means, and variances
        for layer_features, mu, sigma2 in zip(log_features, means, variances):
            # # Ensure mu and sigma2 are tensors and have the correct shape
            # mu = mu.to(layer_features.device)
            # sigma2 = sigma2.to(layer_features.device)

            # Move all tensors to the same device (self.device)
            layer_features = layer_features.to(self.device)
            mu = mu.to(self.device)
            sigma2 = sigma2.to(self.device)
            # Reshape sigma2 to allow broadcasting
            if sigma2.dim() == 1:
                sigma2 = sigma2.view(1, -1)  # Shape (1, num_features)

            # Compute the negative log-likelihood for the layer
            epsilon = 1e-10
            nll = 0.5 * torch.sum(((layer_features - mu) ** 2) / (sigma2+epsilon) + torch.log(2 * torch.pi * (sigma2+epsilon)), dim=0)
            ood_scores.append(nll)

            # # Compute the z=score for the layer
            # epsilon = 1e-10
            # z = torch.sum((layer_features - mu) / (torch.sqrt(sigma2)+epsilon), dim=0)
            # ood_scores.append(z)

        # return torch.sum(torch.stack(ood_scores))

        # TODO: Sum up only scores of those layers that are among max (some % of layers) layers
        # Sum OOD scores for parts
        part1_scores = torch.sum(torch.stack(ood_scores[:385]))
        part2_scores = torch.sum(torch.stack(ood_scores[385:769]))
        part3_scores = torch.sum(torch.stack(ood_scores[769:])) # :1353

        return torch.stack([part1_scores, part2_scores, part3_scores])

    def ood_score(self, new_samples, means, variances):
        """ Compute the OOD score for the new samples using the Gaussian negative log-likelihood method """
        
        # Compute the gradient features for the new batch
        features, num_features, features_scalar = self.gradient_features(new_samples)

        # Take the log of each feature in the batch
        log_features = [torch.log(f + 1e-10) for f in features]  # Add small value to avoid log(0)
        # print(f"{log_features= }")

        # Compute OOD score using Gaussian negative log-likelihood
        ood_scores = self.gaussian_negative_log_likelihood(log_features, means, variances)

        return ood_scores, features, num_features, features_scalar
