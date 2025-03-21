""" Training script for Glow"""

import os
import random
from itertools import islice

import torch
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Loss, RunningAverage
from torch import optim
from torch.utils import data

from models.glow_model.model import Glow
from utilities.utils import to_dataset_wrapper, dataset_names
from utilities.routes import DATAROOT, OUTPUT_DIR

import argparse
import json
import shutil



def check_manual_seed(seed):
    """Check if a manual seed is provided and set it."""
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Using seed: {seed}")

def get_ds_params(dataset_name, dataroot, transform=None, augment=True, download=True, mode='train'):
    """Check if the dataset is valid and return its details."""
    
    if dataset_name in to_dataset_wrapper:
        # Retrieve the dataset wrapper class
        dataset_wrapper = to_dataset_wrapper[dataset_name]

        if mode == "train":
            # Call the get_all method on the wrapper class
            input_size, num_classes, train_dataset, test_dataset = dataset_wrapper.get_all(
                dataroot, transform=transform, augment=augment, download=download)
        elif mode == "test":
            # Call the get_test method on the wrapper class
            input_size, num_classes, train_dataset, test_dataset = dataset_wrapper.get_val(
                dataroot, transform=transform, download=download)
        else:
            raise ValueError(f"Unrecognized mode: {mode}")

    else:
        raise ValueError(f"Unrecognized dataset: {dataset_name}")

    return input_size, num_classes, train_dataset, test_dataset

def compute_loss(nll, reduction="mean"):
    """Compute the loss."""
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}
    losses["total_loss"] = losses["nll"]

    return losses


def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction="mean"):
    """Compute the loss with class condition."""
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    if multi_class:
        y_logits = torch.sigmoid(y_logits)
        loss_classes = F.binary_cross_entropy_with_logits(
            y_logits, y, reduction=reduction
        )
    else:
        loss_classes = F.cross_entropy(
            y_logits, torch.argmax(y, dim=1), reduction=reduction
        )

    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes

    return losses


def train_glow(
    dataset,
    dataroot,
    download,
    augment,
    batch_size,
    eval_batch_size,
    epochs,
    saved_model,
    seed,
    hidden_channels,
    K,
    L,
    actnorm_scale,
    flow_permutation,
    flow_coupling,
    LU_decomposed,
    learn_top,
    y_condition,
    y_weight,
    max_grad_clip,
    max_grad_norm,
    lr,
    n_workers,
    cuda,
    n_init_batches,
    output_dir,
    saved_optimizer,
    warmup,
):
    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
    transform = None
    check_manual_seed(seed)

    ds = get_ds_params(dataset, dataroot, transform, augment, download, mode='train')
    image_shape, num_classes, train_dataset, test_dataset = ds

    # Note: unsupported for now
    multi_class = False

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
    )

    model = Glow(
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        num_classes,
        learn_top,
        y_condition,
    )

    model = model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=5e-5)

    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup)  # noqa
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda)

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x = x.to(device)

        if y_condition:
            y = y.to(device)
            z, nll, y_logits = model(x, y)
            losses = compute_loss_y(nll, y_logits, y_weight, y, multi_class)
        else:
            z, nll, y_logits = model(x, None)
            losses = compute_loss(nll)

        losses["total_loss"].backward()

        if max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        return losses

    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        x = x.to(device)

        with torch.no_grad():
            if y_condition:
                y = y.to(device)
                z, nll, y_logits = model(x, y)
                losses = compute_loss_y(
                    nll, y_logits, y_weight, y, multi_class, reduction="none"
                )
            else:
                z, nll, y_logits = model(x, None)
                losses = compute_loss(nll, reduction="none")

        return losses

    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(
        output_dir, "glow", n_saved=250, require_empty=False
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        checkpoint_handler,
        {"model": model, "optimizer": optimizer},
    )

    monitoring_metrics = ["total_loss"]
    RunningAverage(output_transform=lambda x: x["total_loss"]).attach(
        trainer, "total_loss"
    )

    evaluator = Engine(eval_step)

    # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
    Loss(
        lambda x, y: torch.mean(x),
        output_transform=lambda x: (
            x["total_loss"],
            torch.empty(x["total_loss"].shape[0]),
        ),
    ).attach(evaluator, "total_loss")

    if y_condition:
        monitoring_metrics.extend(["nll"])
        RunningAverage(output_transform=lambda x: x["nll"]).attach(
            trainer, "nll")

        # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
        Loss(
            lambda x, y: torch.mean(x),
            output_transform=lambda x: (
                x["nll"], torch.empty(x["nll"].shape[0])),
        ).attach(evaluator, "nll")

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    # load pre-trained model if given
    if saved_model:
        model.load_state_dict(torch.load(saved_model)["model"])
        model.set_actnorm_init()

        print("model loaded")

        if saved_optimizer:
            optimizer.load_state_dict(torch.load(saved_optimizer))

        file_name, ext = os.path.splitext(saved_model)
        resume_epoch = int(file_name.split("_")[-1])

        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.epoch = resume_epoch
            engine.state.iteration = resume_epoch * len(engine.state.dataloader)

    @trainer.on(Events.STARTED)
    def init(engine):
        model.train()

        init_batches = []
        init_targets = []

        with torch.no_grad():
            for batch, target in islice(train_loader, None, n_init_batches):
                init_batches.append(batch)
                init_targets.append(target)

            init_batches = torch.cat(init_batches).to(device)

            assert init_batches.shape[0] == n_init_batches * batch_size

            if y_condition:
                init_targets = torch.cat(init_targets).to(device)
            else:
                init_targets = None

            model(init_batches, init_targets)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(engine):
        evaluator.run(test_loader)

        scheduler.step()
        metrics = evaluator.state.metrics

        losses = ", ".join(
            [f"{key}: {value:.2f}" for key, value in metrics.items()])

        print(f"Validation Results - Epoch: {engine.state.epoch} {losses}")

    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message(
            f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]"
        )
        timer.reset()

    trainer.run(train_loader, epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=dataset_names,
        help="Type of the dataset to be used.",
    )

    parser.add_argument("--dataroot", type=str, default=DATAROOT, help="path to dataset")

    parser.add_argument("--download", action="store_true", help="downloads dataset")

    parser.add_argument(
        "--no_augment",
        action="store_false",
        dest="augment",
        help="Augment training data",
    )

    parser.add_argument(
        "--hidden_channels", type=int, default=512, help="Number of hidden channels"
    )

    parser.add_argument("--K", type=int, default=32, help="Number of layers per block")

    parser.add_argument("--L", type=int, default=3, help="Number of blocks")

    parser.add_argument(
        "--actnorm_scale", type=float, default=1.0, help="Act norm scale"
    )

    parser.add_argument(
        "--flow_permutation",
        type=str,
        default="invconv",
        choices=["invconv", "shuffle", "reverse"],
        help="Type of flow permutation",
    )

    parser.add_argument(
        "--flow_coupling",
        type=str,
        default="affine",
        choices=["additive", "affine"],
        help="Type of flow coupling",
    )

    parser.add_argument(
        "--no_LU_decomposed",
        action="store_false",
        dest="LU_decomposed",
        help="Train with LU decomposed 1x1 convs",
    )

    parser.add_argument(
        "--no_learn_top",
        action="store_false",
        help="Do not train top layer (prior)",
        dest="learn_top",
    )

    parser.add_argument(
        "--y_condition", action="store_true", help="Train using class condition"
    )

    parser.add_argument(
        "--y_weight", type=float, default=0.01, help="Weight for class condition loss"
    )

    parser.add_argument(
        "--max_grad_clip",
        type=float,
        default=0,
        help="Max gradient value (clip above - for off)",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0,
        help="Max norm of gradient (clip above - 0 for off)",
    )

    parser.add_argument(
        "--n_workers", type=int, default=6, help="number of data loading workers"
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size used during training"
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="batch size used during evaluation",
    )

    parser.add_argument(
        "--epochs", type=int, default=250, help="number of epochs to train for"
    )

    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    parser.add_argument(
        "--warmup",
        type=float,
        default=5,
        help="Use this number of epochs to warmup learning rate linearly from zero to learning rate",  # noqa
    )

    parser.add_argument(
        "--n_init_batches",
        type=int,
        default=8,
        help="Number of batches to use for Act Norm initialisation",
    )

    parser.add_argument(
        "--no_cuda", action="store_false", dest="cuda", help="Disables cuda"
    )

    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Directory to output logs and model checkpoints",
    )

    parser.add_argument(
        "--fresh", action="store_true", help="Remove output directory before starting"
    )

    parser.add_argument(
        "--saved_model",
        default="",
        help="Path to model to load for continuing training",
    )

    parser.add_argument(
        "--saved_optimizer",
        default="",
        help="Path to optimizer to load for continuing training",
    )

    parser.add_argument("--seed", type=int, default=0, help="manual seed")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, f"glow_{args.dataset}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except FileExistsError:
        if args.fresh:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        if (not os.path.isdir(output_dir)) or (
            len(os.listdir(output_dir)) > 0
        ):
            raise FileExistsError(
                "Please provide a path to a non-existing or empty directory. Alternatively, pass the --fresh flag."  # noqa
            )

    kwargs = vars(args)
    del kwargs["fresh"]

    with open(os.path.join(output_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    train_glow(**kwargs)
