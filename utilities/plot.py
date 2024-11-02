""" This script reads OOD scores from files, plots histograms, and calculates AUROC for each test dataset against the fit dataset. """
import argparse
import re
from collections import defaultdict
from os import listdir, makedirs, path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from utilities.utils import dataset_names
from utilities.routes import OUTPUT_DIR


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
def read_ood_scores(file_path):
    """Read OOD scores from a file"""
    # file_path = path.join(data_path, file_name)
    if path.exists(file_path):
        print(f"File '{file_path}' is loading...")
        ood_scores = torch.load(file_path)
        return torch.cat([score.unsqueeze(0) for score in ood_scores], dim=0).cpu().detach().numpy()
    else:
        print(f"File '{file_path}' not found.")
        return None


# Function to plot histograms for fit and test scores
def plot_histogram(fit_scores, test_scores_dict, fit_dataset_name, checkpoint, output_dir):
    """Plot histograms for fit and test scores"""
    # Calculate optimal bins
    test_scores_list = [scores for scores in test_scores_dict.values()]
    bins = best_bin_size(fit_scores, test_scores_list)
    
    plt.figure(figsize=(10, 6))
    
    # Define a more distinctive list of colors for the datasets
    colors = ['#ADD8E6', '#90EE90', '#FFFF00', '#FFC0CB']  # Light blue, light green, yellow, pink
    color_cycle = iter(colors)  # Create an iterator to cycle through the colors
    
    # Plot histogram for fit samples
    plt.hist(fit_scores, bins=bins, alpha=0.7, label=f'Fit Samples ({fit_dataset_name})', color='#8B0000', edgecolor='black')
    
    # Plot histogram for each test dataset with distinct colors
    for test_name, scores in test_scores_dict.items():
        plt.hist(scores, bins=bins, alpha=0.5, label=f'Test Samples ({test_name})', color=next(color_cycle), edgecolor='black')
    
    # Add labels and title
    plt.xlabel('OOD Scores')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of OOD Scores for Test and Fit Samples (Checkpoint {checkpoint})')
    plt.legend(title=f'Bins: {bins}')
    plt.grid(True)

    # Save the plot
    plot_filename = f"histogram_{fit_dataset_name}_checkpoint_{checkpoint}.png"
    plt.savefig(path.join(output_dir, plot_filename))
    
    # Show the plot
    plt.show()

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
        # Read fit scores
        fit_filename = files['fit']
        fit_info = parse_filename(fit_filename)
        fit_dataset_name = fit_info[2]
        f_path = path.join(file_path, fit_filename)
        fit_scores = read_ood_scores(f_path)

        # Read test scores for each test dataset in this checkpoint group
        test_scores_dict = {}
        # test_scores_list = []
        for test_filename in files['tests']:
            test_info = parse_filename(test_filename)
            test_dataset_name = test_info[2]
            f_path = path.join(file_path, test_filename)
            test_scores = read_ood_scores(f_path)
            test_scores_dict[test_dataset_name] = test_scores
            # test_scores_list.append(test_scores)


        # Plot histograms
        plot_histogram(fit_scores, test_scores_dict, fit_dataset_name, checkpoint, output_plot_dir)

        # Plot AUROC
        auroc_df = plot_auroc(fit_scores, test_scores_dict, fit_dataset_name, checkpoint, output_plot_dir)
        save_auroc_csv(auroc_df, output_plot_dir)


def plot_auroc(fit_scores, test_scores_dict, fit_dataset_name, checkpoint, output_dir):
    """Calculate and plot AUROC for fit against each test dataset, returning a DataFrame with AUROC scores."""
    
    # Define the column name based on fit_dataset_name and checkpoint
    column_name = f"{fit_dataset_name}_checkpoint_{checkpoint}"
    
    # Dictionary to store current checkpoint AUROC scores
    auroc_scores = {}

    for test_name, test_scores in test_scores_dict.items():
        # Create labels: 0 for fit samples (in-distribution), 1 for test samples (out-of-distribution)
        labels_fit = np.zeros_like(fit_scores)  # 0 for in-distribution
        labels_test = np.ones_like(test_scores)  # 1 for out-of-distribution

        # Concatenate scores and labels
        all_scores = np.concatenate([fit_scores, test_scores])
        all_labels = np.concatenate([labels_fit, labels_test])

        # Step 1: Compute ROC curve and AUROC
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        auroc = roc_auc_score(all_labels, all_scores)
        
        # Store the AUROC score in the dictionary
        auroc_scores[test_name] = auroc
        
        print(f"AUROC for {test_name} against {fit_dataset_name} (Checkpoint {checkpoint}): {auroc:.4f}")

        # Step 2: Compute Youden's J statistic to find the best threshold
        J_scores = tpr - fpr  # Youden's J statistic
        best_threshold_index = np.argmax(J_scores)  # Index of the best threshold
        best_threshold = thresholds[best_threshold_index]

        # Step 3: Evaluate using the best threshold
        predictions = (all_scores > best_threshold).astype(int)  # Apply the best threshold
        accuracy = accuracy_score(all_labels, predictions)
        conf_matrix = confusion_matrix(all_labels, predictions)
        conf_matrix_text = f"Confusion Matrix:\nTP: {conf_matrix[1, 1]}, FP: {conf_matrix[0, 1]}\nFN: {conf_matrix[1, 0]}, TN: {conf_matrix[0, 0]}"

        print(f"Accuracy using best threshold: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        # Step 4: Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.text(0.6, 0.2, conf_matrix_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8), transform=plt.gca().transAxes)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUROC = {auroc:.4f})")
        plt.scatter(fpr[best_threshold_index], tpr[best_threshold_index], color='red', label=f"Best Threshold (Youden's J) = {best_threshold:.4f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Classifier")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(f"Receiver Operating Characteristic (ROC) Curve for {test_name} vs {fit_dataset_name} (Checkpoint {checkpoint})")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_filename = f'roc_curve_{fit_dataset_name}_{test_name}_checkpoint_{checkpoint}.png'
        plt.savefig(path.join(output_dir, plot_filename))
        plt.show()
    
    # Convert auroc_scores to a DataFrame with one column for the current checkpoint
    checkpoint_df = pd.DataFrame.from_dict(auroc_scores, orient='index', columns=[column_name])

    return checkpoint_df


def save_auroc_csv(auroc_df, output_dir):
    """Save the accumulated AUROC DataFrame to CSV without overwriting previous data."""
    auroc_csv_filename = 'combined_auroc_scores.csv'
    output_path = path.join(output_dir, auroc_csv_filename)
    
    # If the file exists, load the existing data and concatenate
    if path.exists(output_path):
        existing_df = pd.read_csv(output_path, index_col=0)
        auroc_df = pd.concat([existing_df, auroc_df], axis=1)
    
    # Save the combined DataFrame
    auroc_df.to_csv(output_path)
    print(f"Combined AUROC scores saved to {output_path}")


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

    args = parser.parse_args()

    fit_dataset_name = args.fit_dataset
    model_type = args.model_type
    ood_batch_size = int(args.ood_batch_size)

    data_path = path.join(OUTPUT_DIR, f"{model_type}_{fit_dataset_name}")
    d_path = path.join(data_path, ood_batch_size)
    process_and_plot(d_path)
# Run the script independently
if __name__ == "__main__":
    main()
