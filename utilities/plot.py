""" This script reads OOD scores from files, plots histograms, and calculates AUROC for each test dataset against the fit dataset. """
import argparse
import re
from collections import defaultdict
from os import listdir, makedirs, path

from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from utilities.routes import OUTPUT_DIR
from utilities.utils import dataset_names


# Function to extract checkpoint, dataset name, and batch size from the filename
def parse_filename(filename):
    """Parse the filename to extract checkpoint, dataset name, and batch size."""
    # Extract checkpoint number
    checkpoint_match = re.search(r"checkpoint_(\d+)\.pth", filename)
    checkpoint = int(checkpoint_match.group(1)) if checkpoint_match else None

    # Extract dataset names and batch size for fit and test samples
    fit_match = re.search(r"ood_scores_fit_samples_b(\d+)_([a-zA-Z0-9]+)_using_checkpoint", filename)
    test_match = re.search(r"ood_scores_test_samples_b(\d+)_([a-zA-Z0-9]+)_on_([a-zA-Z0-9]+)_using_checkpoint", filename)

    if fit_match:
        batch_size = int(fit_match.group(1))
        dataset_name = fit_match.group(2)  
        return 'fit', batch_size, dataset_name, checkpoint
    
    elif test_match:
        batch_size = int(test_match.group(1))
        test_dataset_name = test_match.group(2)  
        fit_dataset_name = test_match.group(3)   
        return 'test', batch_size, test_dataset_name, fit_dataset_name, checkpoint
    
    return None

# Function to group files by checkpoint
def group_files_by_checkpoint(file_path):
    """Group files by checkpoint number"""
    grouped_files = defaultdict(lambda: {'fit': None, 'tests': []})
    
    file_list = [f for f in listdir(file_path) if (f.startswith('ood_scores') and f.endswith('.pth'))]
    if not file_list:
        raise FileNotFoundError(f"No OOD score files found in {file_path}, Run main to generate OOD scores.")
    for filename in file_list:
        parsed = parse_filename(filename)
        if parsed:
            if parsed[0] == 'fit':  # It's a fit file
                checkpoint = parsed[3]
                grouped_files[checkpoint]['fit'] = filename
            elif parsed[0] == 'test':  # It's a test file
                checkpoint = parsed[4]
                grouped_files[checkpoint]['tests'].append(filename)
    
    # Filter out incomplete groups (i.e., those without both fit and tests)
    return {k: v for k, v in grouped_files.items() if v['fit'] and len(v['tests']) == 4}


# Function to read OOD scores from a file
def read_properties(file_path: str):
    """Read OOD scores from a file and concatenate them for each block."""
    if path.exists(file_path):
        print(f"File '{file_path}' is loading...")
        checkpoint = torch.load(file_path)
        ood_scores = checkpoint['ood_scores']
        features = checkpoint['features']
        num_features = checkpoint['num_features']
        features_scalar = checkpoint['features_scalar']

        # Initialize lists to store concatenated OOD scores for each block
        block1_oods = []
        block2_oods = []
        block3_oods = []

        # Process each layer's OOD score and split them by block
        for score in ood_scores:
            # Assuming 'score' is a tensor of shape [3] with OOD scores for each block
            block1_oods.append(score[0].unsqueeze(0))  # OOD score for block 1
            block2_oods.append(score[1].unsqueeze(0))  # OOD score for block 2
            block3_oods.append(score[2].unsqueeze(0))  # OOD score for block 3

        # Concatenate OOD scores for each block
        block1_oods = torch.cat(block1_oods, dim=0).cpu().detach().numpy()
        block2_oods = torch.cat(block2_oods, dim=0).cpu().detach().numpy()
        block3_oods = torch.cat(block3_oods, dim=0).cpu().detach().numpy()
        all_blocks = [block1_oods, block2_oods, block3_oods]
        # Return the concatenated OOD scores for each block
        return all_blocks, features, num_features, features_scalar

    print(f"File '{file_path}' not found.")
    return None

# # Function to read OOD scores from a file
# def read_properties(file_path: str):
#     """Read OOD scores from a file"""
#     # file_path = path.join(data_path, file_name)
#     if path.exists(file_path):
#         print(f"File '{file_path}' is loading...")
#         checkpoint = torch.load(file_path)
#         ood_scores = checkpoint['ood_scores']
#         features = checkpoint['features']
#         num_features = checkpoint['num_features']
#         features_scalar = checkpoint['features_scalar']
#         ood_score = torch.cat([score.unsqueeze(0) for score in ood_scores], dim=0).cpu().detach().numpy()
#         return ood_score, features, num_features, features_scalar
#     print(f"File '{file_path}' not found.")
#     return None


# Function to plot histograms for fit and test scores
def plot_histogram(fit_scores, test_scores_dict, fit_dataset_name, checkpoint, output_dir):
    """Plot histograms for fit and test scores"""
    # Calculate optimal bins
    test_scores_list = [scores for scores in test_scores_dict.values()]
    # bins = best_bin_size(fit_scores, test_scores_list)
    bins = 25
    plt.figure(figsize=(10, 6))
    
    # Define a more distinctive list of colors for the datasets
    colors = ['#002CFF', '#00FF11', '#F0FF00', '#00FAFF'] # Blue, Green, Yellow, Cyan
    color_cycle = iter(colors)
    
    # Plot histogram for each test dataset with distinct colors
    for test_name, scores in test_scores_dict.items():
        scores = np.concatenate(scores) # Concatenate scores of three blocks
        plt.hist(scores, bins=bins, alpha=0.5, label=f'Test Samples ({test_name})', color=next(color_cycle))#, edgecolor='black')

    # Plot histogram for fit samples
    fit_scores = np.concatenate(fit_scores) # Concatenate scores of three blocks
    plt.hist(fit_scores, bins=bins, alpha=0.7, label=f'Fit Samples ({fit_dataset_name})', color='#FF0000')#, edgecolor='black') # Red
    
    # Add labels and title
    plt.xlabel('OOD Scores')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of OOD Scores Trained on {fit_dataset_name.upper()}')# for Test and Fit Samples (Checkpoint {checkpoint})')
    plt.legend(title=f'Bins: {bins}')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_filename = f"histogram_{fit_dataset_name}_checkpoint_{checkpoint}.png"
    plt.savefig(path.join(output_dir, plot_filename))
    
    # Show the plot
    # plt.show()



def best_bin_size(fit_scores, test_scores_list):
    """Calculate the optimal bin size using the Freedman-Diaconis rule"""
    all_scores = np.concatenate([fit_scores, *test_scores_list])
    q25, q75 = np.percentile(all_scores, [25, 75])
    bin_width = 2 * (q75 - q25) * len(all_scores) ** (-1/3)
    bins = int((all_scores.max() - all_scores.min()) / bin_width)
    return max(bins, 10)  # Ensure a minimum of 10 bins


def process_and_plot(file_path):
    """Process files, read scores, and plot histograms and AUROC"""
    output_plot_dir = path.join(file_path, "plots")

    # Create output directory if it doesn't exist
    makedirs(output_plot_dir, exist_ok=True)

    # Group files by checkpoint
    grouped_files = group_files_by_checkpoint(file_path)
    
    for checkpoint, files in grouped_files.items():
        features_dict = {}
        # Read fit scores
        fit_filename = files['fit']
        fit_info = parse_filename(fit_filename)
        fit_dataset_name = fit_info[2]
        f_path = path.join(file_path, fit_filename)
        fit_scores, fit_features, num_gradients_in_layers, fit_features_scalar = read_properties(f_path)

        # plot_percentile_heatmap(fit_dataset_name, test_dataset_name=None, features=features_scalar, num_gradients_in_layers=num_gradients_in_layers, percentiles = [25, 50, 75, 95, 99], checkpoint=checkpoint, output_dir=output_plot_dir)
        # features_dict[fit_dataset_name] = features_scalar # for layerwise norm plot
        # Read test scores for each test dataset in this checkpoint group
        test_scores_dict = {}
        for test_filename in files['tests']:
            test_info = parse_filename(test_filename)
            test_dataset_name = test_info[2]
            f_path = path.join(file_path, test_filename)
            test_scores, test_features, num_gradients_in_layers, features_scalar = read_properties(f_path)
            test_scores_dict[test_dataset_name] = test_scores
            # features_dict[test_dataset_name] = features_scalar # for layerwise norm plot
            features_dict[test_dataset_name] = features_scalar
            # plot_percentile_heatmap(fit_dataset_name, test_dataset_name, features_scalar, num_gradients_in_layers, percentiles = [25, 50, 75, 95, 99], checkpoint=checkpoint, output_dir=output_plot_dir)

        # Plot layer-wise normalized parameter values with smoothed curves, using averaging window
        # plot_layerwise_norm(features_dict, num_gradients_in_layers, checkpoint, output_plot_dir)
        # Plot histograms
        plot_histogram(fit_scores, test_scores_dict, fit_dataset_name, checkpoint, output_plot_dir)

        # # Plot AUROC
        auroc_df = plot_auroc_subplot(fit_scores, test_scores_dict, fit_dataset_name, checkpoint, output_plot_dir) # AUROC using scores
        # auroc_df = plot_auroc_subplot(fit_features_scalar, features_dict, fit_dataset_name, checkpoint, output_plot_dir, num_gradients_in_layers) # AUROC using Squared L2 norms
        save_auroc_csv(auroc_df, output_plot_dir)


def plot_layerwise_norm(features_dict, num_gradients_in_layers, checkpoint, output_dir):
    """
    Plot layer-wise normalized parameter values for fit and test datasets.
    
    Args:
    - features_dict: dict, {dataset_name: features_scalar}
    - num_gradients_in_layers: list, gradient counts for each layer
    - checkpoint: str, checkpoint identifier for labeling
    - output_dir: str, directory to save the plot
    """
    # Define the layers
    layers = range(1, len(num_gradients_in_layers) + 1)
    
    # Create the plot using averraging window technique
    plt.figure(figsize=(12, 8))
    color_palette = ['blue', 'green', 'red', 'orange', 'purple']
    for i, (dataset_name, features_scalar) in enumerate(features_dict.items()):
        normalized_features = [f / n for f, n in zip(features_scalar, num_gradients_in_layers)]
        smoothed_features = uniform_filter1d(normalized_features, size=200)  # Smooth with a window
        plt.plot(
            layers, 
            smoothed_features, 
            label=f'{dataset_name}', 
            color=color_palette[i % len(color_palette)], 
            linewidth=2
        )

    # # Plot the layer-wise normalized parameter values for each dataset
    # for i, (dataset_name, features_scalar) in enumerate(features_dict.items()):
    #     # Normalize the features based on num_gradients_in_layers
    #     normalized_features = [f / n for f, n in zip(features_scalar, num_gradients_in_layers)]
    #     normalized_features = np.log1p(normalized_features)  # Using log1p for numerical stability
    #     # Plot each dataset with a unique color and label
    #     plt.plot(
    #         layers,
    #         normalized_features,
    #         label=f'{dataset_name}',
    #         color=color_palette[i % len(color_palette)],
    #         marker='o',
    #         linewidth=2
    #     )
    
    # Label the axes and title
    plt.xlabel('Layer Number', fontsize=14)
    plt.ylabel('Parameter-Normalized Values', fontsize=14)
    plt.title(f'Layerwise Parameter-Normalized Values - Checkpoint {checkpoint}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plot_filename = f'layerwise_norm_checkpoint{checkpoint}.png'
    plt.savefig(path.join(output_dir, plot_filename))
    plt.close()



# Plot AUROC for fit against each test dataset using subplots according to the given percentiles using 3 different saved blocks of scores
def plot_auroc_subplot(fit_scores, test_scores_dict, fit_dataset_name, checkpoint, output_dir):
    """Calculate and plot AUROC for fit against each test dataset using subplots, returning a DataFrame with AUROC scores."""
    percentiles = [0, 25, 50, 75, 95, 99]  # List of percentiles to process
    auroc_dfs = []  # List to collect AUROC DataFrames for each percentile

    for percentile in percentiles:
        # Define the column name based on fit_dataset_name and checkpoint
        column_name = f"{fit_dataset_name}_checkpoint_{checkpoint}_per_{percentile}"
        
        # Dictionary to store current checkpoint AUROC scores
        auroc_scores = {}

        # Define the number of test datasets for subplots
        num_tests = len(test_scores_dict)
        fig, axs = plt.subplots(1, num_tests, figsize=(5 * num_tests, 6), constrained_layout=True)

        # Ensure axs is iterable even when there's only one subplot
        if num_tests == 1:
            axs = [axs]
        fit_scores_selected = np.concatenate(fit_scores)
        # fit_scores_selected = fit_scores[2]
        # Filter fit scores based on the current percentile
        fit_threshold = np.percentile(fit_scores_selected, percentile)
        fit_filtered_features = [f for f in fit_scores_selected if f > fit_threshold]

        for ax, (test_name, test_scores) in zip(axs, test_scores_dict.items()):
            test_scores_selected = np.concatenate(test_scores)
            # test_scores_selected = test_scores[2]
            # Filter test scores based on the current percentile
            test_threshold = np.percentile(test_scores_selected, percentile)
            test_filtered_features = [f for f in test_scores_selected if f > test_threshold]

            # Create labels: 0 for fit samples, 1 for test samples
            labels_fit = np.zeros(len(fit_filtered_features))  # 0 for in-distribution
            labels_test = np.ones(len(test_filtered_features))  # 1 for out-of-distribution

            # Concatenate scores and labels
            all_scores = np.concatenate([fit_filtered_features, test_filtered_features])
            all_labels = np.concatenate([labels_fit, labels_test])

            # Compute ROC curve and AUROC
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            auroc = roc_auc_score(all_labels, all_scores)

            # Store the AUROC score in the dictionary
            auroc_scores[test_name] = auroc
            
            print(f"AUROC for {test_name} against {fit_dataset_name} (Checkpoint {checkpoint}, {percentile}%): {auroc:.4f}")

            # Compute Youden's J statistic to find the best threshold
            J_scores = tpr - fpr
            best_threshold_index = np.argmax(J_scores)
            best_threshold = thresholds[best_threshold_index]

            # Evaluate using the best threshold
            predictions = (all_scores > best_threshold).astype(int)
            accuracy = accuracy_score(all_labels, predictions)
            conf_matrix = confusion_matrix(all_labels, predictions)
            conf_matrix_text = f"Confusion Matrix:\nTP: {conf_matrix[1, 1]}, FP: {conf_matrix[0, 1]}\nFN: {conf_matrix[1, 0]}, TN: {conf_matrix[0, 0]}"

            print(f"Accuracy using best threshold: {accuracy:.4f}")
            print(f"Confusion Matrix:\n{conf_matrix}")

            # Plot ROC curve using RocCurveDisplay
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc, estimator_name=test_name)
            display.plot(ax=ax)

            # Highlight the best threshold on the plot
            ax.scatter(fpr[best_threshold_index], tpr[best_threshold_index], color='#FF0000', 
                    label=f"Best Threshold (Youden's J) = {best_threshold:.4f}")
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Classifier")

            # Set labels, title, and legend
            ax.set_title(f"ROC for {fit_dataset_name} vs {test_name}")
            ax.set_xlabel("False Positive Rate (FPR)")
            ax.set_ylabel("True Positive Rate (TPR)")
            ax.legend(loc='lower right')

            # Display the confusion matrix text
            ax.text(0.6, 0.2, conf_matrix_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8), transform=ax.transAxes)
            test_scores_selected = None

        # Set the main title for the entire figure
        fig.suptitle(f"ROC Curves for {fit_dataset_name} (Checkpoint {checkpoint}), {percentile}%", fontsize=16)

        # Save the entire figure
        plot_filename = f'roc_curves_{fit_dataset_name}_checkpoint_{checkpoint}_per_{percentile}.png'
        plt.savefig(path.join(output_dir, plot_filename))
        plt.show()

        # Convert auroc_scores to a DataFrame with one column for the current checkpoint
        checkpoint_df = pd.DataFrame.from_dict(auroc_scores, orient='index', columns=[column_name])
        auroc_dfs.append(checkpoint_df)

    # Combine all AUROC DataFrames across percentiles
    combined_auroc_df = pd.concat(auroc_dfs, axis=1)

    return combined_auroc_df





# # Plot AUROC for fit against each test dataset using subplots according to the given percentiles
# def plot_auroc_subplot(fit_scores, test_scores_dict, fit_dataset_name, checkpoint, output_dir, num_gradients_in_layers):
#     """Calculate and plot AUROC for fit against each test dataset using subplots, returning a DataFrame with AUROC scores."""
#     percentiles = [0, 25, 50, 75, 95, 99]  # List of percentiles to process
#     auroc_dfs = []  # List to collect AUROC DataFrames for each percentile
    
#     normalized_fit = [f / n for f, n in zip(fit_scores, num_gradients_in_layers)]
#     fit_scores = np.log1p(normalized_fit)


#     for percentile in percentiles:
#         # Define the column name based on fit_dataset_name and checkpoint
#         column_name = f"{fit_dataset_name}_checkpoint_{checkpoint}_per_{percentile}"
        
#         # Dictionary to store current checkpoint AUROC scores
#         auroc_scores = {}

#         # Define the number of test datasets for subplots
#         num_tests = len(test_scores_dict)
#         fig, axs = plt.subplots(1, num_tests, figsize=(5 * num_tests, 6), constrained_layout=True)

#         # Ensure axs is iterable even when there's only one subplot
#         if num_tests == 1:
#             axs = [axs]

#         # Filter fit scores based on the current percentile
#         fit_threshold = np.percentile(fit_scores, percentile)
#         fit_filtered_features = [f for f in fit_scores if f > fit_threshold]

#         for ax, (test_name, test_scores) in zip(axs, test_scores_dict.items()):

#             normalized_test = [f / n for f, n in zip(test_scores, num_gradients_in_layers)]
#             test_scores = np.log1p(normalized_test)

#             # Filter test scores based on the current percentile
#             test_threshold = np.percentile(test_scores, percentile)
#             test_filtered_features = [f for f in test_scores if f > test_threshold]

#             # Create labels: 0 for fit samples, 1 for test samples
#             labels_fit = np.zeros(len(fit_filtered_features))  # 0 for in-distribution
#             labels_test = np.ones(len(test_filtered_features))  # 1 for out-of-distribution

#             # Concatenate scores and labels
#             all_scores = np.concatenate([fit_filtered_features, test_filtered_features])
#             all_labels = np.concatenate([labels_fit, labels_test])

#             # Compute ROC curve and AUROC
#             fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
#             auroc = roc_auc_score(all_labels, all_scores)

#             # Store the AUROC score in the dictionary
#             auroc_scores[test_name] = auroc
            
#             print(f"AUROC for {test_name} against {fit_dataset_name} (Checkpoint {checkpoint}, {percentile}%): {auroc:.4f}")

#             # Compute Youden's J statistic to find the best threshold
#             J_scores = tpr - fpr
#             best_threshold_index = np.argmax(J_scores)
#             best_threshold = thresholds[best_threshold_index]

#             # Evaluate using the best threshold
#             predictions = (all_scores > best_threshold).astype(int)
#             accuracy = accuracy_score(all_labels, predictions)
#             conf_matrix = confusion_matrix(all_labels, predictions)
#             conf_matrix_text = f"Confusion Matrix:\nTP: {conf_matrix[1, 1]}, FP: {conf_matrix[0, 1]}\nFN: {conf_matrix[1, 0]}, TN: {conf_matrix[0, 0]}"

#             print(f"Accuracy using best threshold: {accuracy:.4f}")
#             print(f"Confusion Matrix:\n{conf_matrix}")

#             # Plot ROC curve using RocCurveDisplay
#             display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc, estimator_name=test_name)
#             display.plot(ax=ax)

#             # Highlight the best threshold on the plot
#             ax.scatter(fpr[best_threshold_index], tpr[best_threshold_index], color='#FF0000', 
#                     label=f"Best Threshold (Youden's J) = {best_threshold:.4f}")
#             ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Classifier")

#             # Set labels, title, and legend
#             ax.set_title(f"ROC for {fit_dataset_name} vs {test_name}")
#             ax.set_xlabel("False Positive Rate (FPR)")
#             ax.set_ylabel("True Positive Rate (TPR)")
#             ax.legend(loc='lower right')

#             # Display the confusion matrix text
#             ax.text(0.6, 0.2, conf_matrix_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8), transform=ax.transAxes)

#         # Set the main title for the entire figure
#         fig.suptitle(f"ROC Curves for {fit_dataset_name} (Checkpoint {checkpoint}), {percentile}%", fontsize=16)

#         # Save the entire figure
#         plot_filename = f'roc_curves_{fit_dataset_name}_checkpoint_{checkpoint}_per_{percentile}.png'
#         plt.savefig(path.join(output_dir, plot_filename))
#         plt.show()

#         # Convert auroc_scores to a DataFrame with one column for the current checkpoint
#         checkpoint_df = pd.DataFrame.from_dict(auroc_scores, orient='index', columns=[column_name])
#         auroc_dfs.append(checkpoint_df)

#     # Combine all AUROC DataFrames across percentiles
#     combined_auroc_df = pd.concat(auroc_dfs, axis=1)

#     return combined_auroc_df


# # Plot AUROC for fit against each test dataset using subplots according to the given percentiles
# def plot_auroc_subplot(fit_scores, test_scores_dict, fit_dataset_name, checkpoint, output_dir):
#     """Calculate and plot AUROC for fit against each test dataset using subplots, returning a DataFrame with AUROC scores."""
#     percentiles = [0, 25, 50, 75, 95, 99]  # List of percentiles to process
#     auroc_dfs = []  # List to collect AUROC DataFrames for each percentile

#     for percentile in percentiles:
#         # Define the column name based on fit_dataset_name and checkpoint
#         column_name = f"{fit_dataset_name}_checkpoint_{checkpoint}_per_{percentile}"
        
#         # Dictionary to store current checkpoint AUROC scores
#         auroc_scores = {}

#         # Define the number of test datasets for subplots
#         num_tests = len(test_scores_dict)
#         fig, axs = plt.subplots(1, num_tests, figsize=(5 * num_tests, 6), constrained_layout=True)

#         # Ensure axs is iterable even when there's only one subplot
#         if num_tests == 1:
#             axs = [axs]

#         # Filter fit scores based on the current percentile
#         fit_threshold = np.percentile(fit_scores, percentile)
#         fit_filtered_features = [f for f in fit_scores if f > fit_threshold]

#         for ax, (test_name, test_scores) in zip(axs, test_scores_dict.items()):

#             # Filter test scores based on the current percentile
#             test_threshold = np.percentile(test_scores, percentile)
#             test_filtered_features = [f for f in test_scores if f > test_threshold]

#             # Create labels: 0 for fit samples, 1 for test samples
#             labels_fit = np.zeros(len(fit_filtered_features))  # 0 for in-distribution
#             labels_test = np.ones(len(test_filtered_features))  # 1 for out-of-distribution

#             # Concatenate scores and labels
#             all_scores = np.concatenate([fit_filtered_features, test_filtered_features])
#             all_labels = np.concatenate([labels_fit, labels_test])

#             # Compute ROC curve and AUROC
#             fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
#             auroc = roc_auc_score(all_labels, all_scores)

#             # Store the AUROC score in the dictionary
#             auroc_scores[test_name] = auroc
            
#             print(f"AUROC for {test_name} against {fit_dataset_name} (Checkpoint {checkpoint}, {percentile}%): {auroc:.4f}")

#             # Compute Youden's J statistic to find the best threshold
#             J_scores = tpr - fpr
#             best_threshold_index = np.argmax(J_scores)
#             best_threshold = thresholds[best_threshold_index]

#             # Evaluate using the best threshold
#             predictions = (all_scores > best_threshold).astype(int)
#             accuracy = accuracy_score(all_labels, predictions)
#             conf_matrix = confusion_matrix(all_labels, predictions)
#             conf_matrix_text = f"Confusion Matrix:\nTP: {conf_matrix[1, 1]}, FP: {conf_matrix[0, 1]}\nFN: {conf_matrix[1, 0]}, TN: {conf_matrix[0, 0]}"

#             print(f"Accuracy using best threshold: {accuracy:.4f}")
#             print(f"Confusion Matrix:\n{conf_matrix}")

#             # Plot ROC curve using RocCurveDisplay
#             display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc, estimator_name=test_name)
#             display.plot(ax=ax)

#             # Highlight the best threshold on the plot
#             ax.scatter(fpr[best_threshold_index], tpr[best_threshold_index], color='#FF0000', 
#                     label=f"Best Threshold (Youden's J) = {best_threshold:.4f}")
#             ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Classifier")

#             # Set labels, title, and legend
#             ax.set_title(f"ROC for {fit_dataset_name} vs {test_name}")
#             ax.set_xlabel("False Positive Rate (FPR)")
#             ax.set_ylabel("True Positive Rate (TPR)")
#             ax.legend(loc='lower right')

#             # Display the confusion matrix text
#             ax.text(0.6, 0.2, conf_matrix_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8), transform=ax.transAxes)

#         # Set the main title for the entire figure
#         fig.suptitle(f"ROC Curves for {fit_dataset_name} (Checkpoint {checkpoint}), {percentile}%", fontsize=16)

#         # Save the entire figure
#         plot_filename = f'roc_curves_{fit_dataset_name}_checkpoint_{checkpoint}_per_{percentile}.png'
#         plt.savefig(path.join(output_dir, plot_filename))
#         plt.show()

#         # Convert auroc_scores to a DataFrame with one column for the current checkpoint
#         checkpoint_df = pd.DataFrame.from_dict(auroc_scores, orient='index', columns=[column_name])
#         auroc_dfs.append(checkpoint_df)

#     # Combine all AUROC DataFrames across percentiles
#     combined_auroc_df = pd.concat(auroc_dfs, axis=1)

#     return combined_auroc_df








# # Plot AUROC for fit against each test dataset using subplots without any percentiles
# def plot_auroc_subplot(fit_scores, test_scores_dict, fit_dataset_name, checkpoint, output_dir):
#     """Calculate and plot AUROC for fit against each test dataset using subplots, returning a DataFrame with AUROC scores."""

    # # Define the column name based on fit_dataset_name and checkpoint
    # column_name = f"{fit_dataset_name}_checkpoint_{checkpoint}"
    
    # # Dictionary to store current checkpoint AUROC scores
    # auroc_scores = {}

    # # Define the number of test datasets for subplots
    # num_tests = len(test_scores_dict)
    # fig, axs = plt.subplots(1, num_tests, figsize=(5 * num_tests, 6), constrained_layout=True)
    
    # # Ensure axs is iterable even when there's only one subplot
    # if num_tests == 1:
    #     axs = [axs]
    # for ax, (test_name, test_scores) in zip(axs, test_scores_dict.items()):
    #     # Create labels: 0 for fit samples (in-distribution), 1 for test samples (out-of-distribution)
    #     labels_fit = np.zeros_like(fit_scores)  # 0 for in-distribution
    #     labels_test = np.ones_like(test_scores)  # 1 for out-of-distribution

    #     # Concatenate scores and labels
    #     all_scores = np.concatenate([fit_scores, test_scores])
    #     all_labels = np.concatenate([labels_fit, labels_test])

    #     # Compute ROC curve and AUROC
    #     fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    #     auroc = roc_auc_score(all_labels, all_scores)
        
    #     # Store the AUROC score in the dictionary
    #     auroc_scores[test_name] = auroc
        
    #     print(f"AUROC for {test_name} against {fit_dataset_name} (Checkpoint {checkpoint}): {auroc:.4f}")

    #     # Compute Youden's J statistic to find the best threshold
    #     J_scores = tpr - fpr
    #     best_threshold_index = np.argmax(J_scores)
    #     best_threshold = thresholds[best_threshold_index]

    #     # Evaluate using the best threshold
    #     predictions = (all_scores > best_threshold).astype(int)
    #     accuracy = accuracy_score(all_labels, predictions)
    #     conf_matrix = confusion_matrix(all_labels, predictions)
    #     conf_matrix_text = f"Confusion Matrix:\nTP: {conf_matrix[1, 1]}, FP: {conf_matrix[0, 1]}\nFN: {conf_matrix[1, 0]}, TN: {conf_matrix[0, 0]}"

    #     print(f"Accuracy using best threshold: {accuracy:.4f}")
    #     print(f"Confusion Matrix:\n{conf_matrix}")

    #     # Plot ROC curve using RocCurveDisplay
    #     display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc, estimator_name=test_name)
    #     display.plot(ax=ax)
        
    #     # Highlight the best threshold on the plot
    #     ax.scatter(fpr[best_threshold_index], tpr[best_threshold_index], color='#FF0000', 
    #                label=f"Best Threshold (Youden's J) = {best_threshold:.4f}")
    #     ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Classifier")
        
    #     # Set labels, title, and legend
    #     ax.set_title(f"ROC for {fit_dataset_name} vs {test_name}")
    #     ax.set_xlabel("False Positive Rate (FPR)")
    #     ax.set_ylabel("True Positive Rate (TPR)")
    #     ax.legend(loc='lower right')
        
    #     # Display the confusion matrix text
    #     ax.text(0.6, 0.2, conf_matrix_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8), transform=ax.transAxes)
    
    # # Set the main title for the entire figure
    # fig.suptitle(f"ROC Curves for {fit_dataset_name} (Checkpoint {checkpoint})", fontsize=16)
    
    # # Save the entire figure
    # plot_filename = f'roc_curves_{fit_dataset_name}_checkpoint_{checkpoint}.png'
    # plt.savefig(path.join(output_dir, plot_filename))
    # plt.show()

    # # Convert auroc_scores to a DataFrame with one column for the current checkpoint
    # checkpoint_df = pd.DataFrame.from_dict(auroc_scores, orient='index', columns=[column_name])

    # return checkpoint_df


def save_auroc_csv(auroc_df, output_dir):
    """Save the accumulated AUROC DataFrame to CSV without overwriting previous data."""
    # auroc_csv_filename = 'combined_auroc_scores.csv'
    auroc_csv_filename = 'combined_auroc_features.csv'
    output_path = path.join(output_dir, auroc_csv_filename)
    
    # If the file exists, load the existing data and concatenate
    if path.exists(output_path):
        existing_df = pd.read_csv(output_path, index_col=0)
        for column in auroc_df.columns:
            if column in existing_df.columns:
                existing_df[column] = auroc_df[column]
            else:
                existing_df = pd.concat([existing_df, auroc_df[[column]]], axis=1)
        auroc_df = existing_df
    
    # Save the combined DataFrame
    auroc_df.to_csv(output_path)
    print(f"Combined AUROC scores saved to {output_path}")


def plot_percentile_heatmap(fit_dataset_name, test_dataset_name, features, num_gradients_in_layers, percentiles = [25, 50, 75, 95, 99], checkpoint="", output_dir=None):
    """Plot a heatmap of the log of normalized layer-wise squared L2 norms above the given percentiles."""
    # Normalize the features based on num_gradients_in_layers
    normalized_features = [f / n for f, n in zip(features, num_gradients_in_layers)]
    # checkpoint_number = int(''.join(filter(str.isdigit, checkpoint)))
    # Apply logarithmic scaling
    log_features = np.log1p(normalized_features)  # Using log1p for numerical stability

    thresholds = np.percentile(log_features, percentiles)
    filtered_features = {p: [(i+1, f) for i, f in enumerate(log_features) if f > t] for p, t in zip(percentiles, thresholds)}

    fig, axs = plt.subplots(len(percentiles), 1, figsize=(15, len(percentiles) * 2), constrained_layout=True)
    
    for ax, (p, feature) in zip(axs, filtered_features.items()):
        if feature:
            layer_positions, filtered_features = zip(*feature)
            heatmap_data = np.array(filtered_features).reshape(1, -1)
            
            sns.heatmap(heatmap_data, cmap="YlGnBu", cbar=True, xticklabels=layer_positions, ax=ax)
            ax.set_title(f'Layers greater than {p}th Percentile')
    
    if test_dataset_name: # "Layerwise $|L|_{{2}}^{{2}}$ Density" - The term "density" here suggests the normalization by the parameter count.
        fig.suptitle(f'Log of Parameter-Normalized Layerwise $|L|_{{2}}^{{2}}$ on {fit_dataset_name} vs. {test_dataset_name} using {checkpoint}', fontsize=16)
        plot_filename = f'percentile_heatmap_{fit_dataset_name}_vs_{test_dataset_name}_checkpoint_{checkpoint}.png'
    else:
        fig.suptitle(f'Log of Parameter-Normalized Layerwise $|L|_{{2}}^{{2}}$ on {fit_dataset_name} using {checkpoint}', fontsize=16)
        plot_filename = f'percentile_heatmap_{fit_dataset_name}_checkpoint_{checkpoint}.png'
    # Save the figure

    # plot_filename = f'percentile_heatmap_{fit_dataset_name}_vs_{test_filename}_checkpoint_{checkpoint}.png'
    plt.savefig(path.join(output_dir, plot_filename))
    plt.show()

    # for p, feature in filtered_features.items():
    #     if feature:
    #         layer_positions, filtered_features = zip(*feature)
    #         heatmap_data = np.array(filtered_features).reshape(1, -1)
            
    #         plt.figure(figsize=(15, 1))
    #         sns.heatmap(heatmap_data, cmap="YlGnBu", cbar=True, xticklabels=layer_positions)
    #         plt.title(f'Log of Normalized Layer-wise Squared L2 Norm on {fit_dataset_name} vs. {test_filename} (Above {p}th Percentile) using {checkpoint}')
    #         plt.show()




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
        choices=dataset_names,#to_dataset_wrapper.keys(),
        help="Dataset to use for fitting."
    )
    parser.add_argument(
        "--ood_batch_size",
        type=str,
        default="1",
        help="Batch size for OOD score computation."
    )
    parser.add_argument(
        "--histogram",
        type=bool,
        default=True,
        help="Whether to plot the histograms."
    )
    parser.add_argument(
        "--auroc",
        type=bool,
        default=True,
        help="Whether to plot the AUROC."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=OUTPUT_DIR,
        help="Path to the data"
    )

    args = parser.parse_args()

    fit_dataset_name = args.fit_dataset
    model_type = args.model_type
    ood_batch_size = args.ood_batch_size
    path_d = args.path

    data_path = path.join(path_d, f"{model_type}_{fit_dataset_name}")
    d_path = path.join(data_path, ood_batch_size)
    process_and_plot(d_path)
# Run the script independently
if __name__ == "__main__":
    main()
