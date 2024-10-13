""" Run the main program. """

import torch
from torch.utils.data import DataLoader

from models.load_models import LoadModels
from ood_scores.features.feature_extractors import FeatureExtractor
from ood_scores.ood_extractors import OodScoresExtractor



if __name__ == '__main__':

    load_models = LoadModels()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TODO: read from input
    num_samples = 1000
    batch_size = 1


    model, fit_train, fit_test, test_train, test_test = load_models.glow(checkpoint="glow_checkpoint_195250.pt", fit_dataset_name='cifar10', test_dataset_name='svhn')

    feature_extractor = FeatureExtractor(model, DEVICE)
    ood_scores_extractor = OodScoresExtractor(model, DEVICE)



    # loading based on fit test, train or combination of both TODO: read input
    fit_loader = DataLoader(fit_test, batch_size=batch_size, shuffle=True)

    # Fit Gaussians to the log of the gradient features
    means, variances, all_features = feature_extractor.fit_gaussians_to_log_features(fit_loader)
    # Compute OOD scores for the test samples
    ood_scores_extractor.ood_scores_on_batches(fit_test, test_test, batch_size, means, variances)
