# ood_detection

First create a folder named `data` under the `root` and put two datasets there. All other datasets will be downloaded.

To train the model on three-channel datasets [cifar10, celeba, imagenet32, gstrb, svhn] just run the following command:

```bash
python3 -m models.glow_model.train --loop_over_all True
```

the checkpoints will be created under the name of each dataset.
