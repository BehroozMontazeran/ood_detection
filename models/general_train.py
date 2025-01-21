import argparse
import json
import shutil
from os import path, makedirs
from multiprocessing import Process

from models.glow_model.train import train_glow  # , train_vae, train_diffusion
from utilities.routes import OUTPUT_DIR, DATAROOT, PathCreator
from utilities.utils import to_dataset_wrapper


def setup_output_dir(output_dir, fresh=False):
    """Set up the output directory."""
    try:
        makedirs(output_dir, exist_ok=True)
        if fresh and path.isdir(output_dir):
            shutil.rmtree(output_dir)
            makedirs(output_dir)
    except FileExistsError as exc:
        raise FileExistsError("Provide a non-existing or empty directory. Use --fresh to overwrite.") from exc


def save_hyperparameters(output_dir, kwarg):
    """Save hyperparameters as JSON."""
    with open(path.join(output_dir, "hparams.json"), "w", encoding="utf-8") as fp:
        json.dump(kwarg, fp, sort_keys=True, indent=4)


def train_model(model_type, **kwargs_):
    """Train a model with specified type on the selected dataset."""
    if model_type == "glow":
        train_glow(**kwargs_)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_on_dataset(dataset_name, model_type, kwargs, fresh_arg, path_creator):
    """Train a single dataset-model combination."""
    # Create a unique sub-directory for each dataset-model combination
    dataset_output_dir = path_creator.model_dataset_path(model_type, dataset_name)
    setup_output_dir(dataset_output_dir, fresh_arg)

    # Prepare dataset-specific arguments
    kwargs["dataset"] = dataset_name
    kwargs["output_dir"] = dataset_output_dir

    # Save hyperparameters
    save_hyperparameters(dataset_output_dir, kwargs)

    # Call train_model for each model-dataset combo
    print(f"Training {model_type} on {dataset_name}...")
    train_model(model_type, **kwargs)


if __name__ == "__main__":
    path_creator = PathCreator()
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--parallel", action="store_true", help="Train datasets in parallel using multiple processes")
    parser.add_argument("--dataset", type=str, default="", choices=to_dataset_wrapper.keys(), help="Dataset type")
    parser.add_argument("--dataroot", type=str, default=DATAROOT, help="Path to datasets root directory")
    parser.add_argument("--download", action="store_true", help="Download dataset if needed")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Base output directory")
    parser.add_argument("--model_type", type=str, default="glow", choices=["glow", "vae", "diffusion"], help="Model type")
    parser.add_argument("--fresh", action="store_true", help="Clear the output directory before starting")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Model-specific arguments (glow)
    parser.add_argument("--no_augment",action="store_false", dest="augment", help="Augment training data",)
    parser.add_argument("--hidden_channels", type=int, default=512, help="Number of hidden channels")
    parser.add_argument("--K", type=int, default=32, help="Number of layers per block")
    parser.add_argument("--L", type=int, default=3, help="Number of blocks")
    parser.add_argument("--actnorm_scale", type=float, default=1.0, help="Act norm scale")
    parser.add_argument("--flow_permutation", type=str, default="invconv",
                        choices=["invconv", "shuffle", "reverse"], help="Type of flow permutation")
    parser.add_argument("--flow_coupling", type=str, default="affine",
                        choices=["additive", "affine"], help="Type of flow coupling")
    parser.add_argument("--no_LU_decomposed", action="store_false", dest="LU_decomposed",
                        help="Train with LU decomposed 1x1 convs")
    parser.add_argument("--no_learn_top", action="store_false", help="Do not train top layer (prior)", dest="learn_top")
    parser.add_argument("--y_condition", action="store_true", help="Train using class condition")
    parser.add_argument("--y_weight", type=float, default=0.01, help="Weight for class condition loss")
    parser.add_argument("--max_grad_clip", type=float, default=0, help="Max gradient value (clip above - for off)")
    parser.add_argument("--max_grad_norm", type=float, default=0, help="Max norm of gradient (clip above - 0 for off)")
    parser.add_argument("--n_workers", type=int, default=6, help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size used during training")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="batch size used during evaluation")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup", type=float, default=5, help="Warmup epochs for learning rate")
    parser.add_argument("--n_init_batches", type=int, default=8, help="Batches for Act Norm init")
    parser.add_argument("--no_cuda", action="store_false", dest="cuda", help="Disable CUDA")
    parser.add_argument("--saved_model", default="", help="Path to model to load for continuing training")
    parser.add_argument("--saved_optimizer", default="", help="Path to optimizer to load for continuing training")



    #TODO: Add model-specific arguments for VAE and Diffusion


    args = parser.parse_args()
    kwargs = vars(args)
    model_type_arg = kwargs.pop("model_type")
    fresh_arg = kwargs.pop("fresh")
    parallel_mode = kwargs.pop("parallel")

    # Set up the output directory
    # setup_output_dir(args.output_dir, fresh_arg)

    if parallel_mode:
        # Train datasets in parallel using multiprocessing
        processes = []
        for dataset_name in to_dataset_wrapper.keys():
            process = Process(target=train_on_dataset, args=(dataset_name, model_type_arg, kwargs, fresh_arg, path_creator))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
    else:
        # Train datasets sequentially
        for dataset_name in to_dataset_wrapper.keys():
            train_on_dataset(dataset_name, model_type_arg, kwargs, fresh_arg, path_creator)
