# ood_detection

First copy two datasets in `data` under `root`. All other datasets will be downloaded.

To train the model on three-channel datasets [cifar10, celeba, imagenet32, gstrb, svhn] in parallel just run the following command

```bash
python3 -m models.general_train --parallel True
```

the checkpoints will be created in seperate folder under the name of each dataset.
