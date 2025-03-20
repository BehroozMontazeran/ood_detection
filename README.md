# OOD Detection with immature Models

This repository is replication of paper `OOD Detection with immature Models` (https://arxiv.org/abs/2502.00820).


### Train Models

To train the model on three-channel datasets [cifar10, celeba, imagenet32, gstrb, svhn] in parallel just run the following command

```bash
python3 -m models.general_train --parallel
```

the checkpoints will be created in seperate folder under the name of each dataset.

### Extract the OOD Scores based on saved Checkpoints

To extract the OOd Scores based on default settings run the following:

```bash
python3 -m main --loop_over_all
```
by which the model would be `glow` and it will loop over all datasets[cifar10, celeba, imagenet32, gstrb, svhn], ood_batch_sizes {1,5} and checkpoints {0, -1}.

At the end of OOD Score calculation the related plots will be saved in each separate folder under the name of `model_dataset_name`.
