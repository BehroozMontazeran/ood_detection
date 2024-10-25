from generative_model import GenerativeModel
from path_definitions import VAE_ROOT
from utilities.utils import to_dataset_wrapper

from os import path

import torch

from torch import nn

import json

# from torchvision import datasets, transforms
from itertools import chain

from data.datasets import get_fashionmnist, get_cifar10, get_svhn, get_celeba, get_imagenet32

import matplotlib.pyplot as plt


class SimpleVAE(nn.Module, GenerativeModel):
    def __init__(self, input_shape=(32, 32, 3), latent_dims=64, encoder=None, decoder=None):
        super().__init__()

        self.input_shape = input_shape
        self.latent_dims = latent_dims

        self.encoder = encoder
        self.decoder = decoder

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # TODO: write this to be non-device specific
        self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        pass

    def get_normal_sample(self, means, stds):
        return means + stds * self.N.sample((self.latent_dims,))

    def eval_nll(self, x):
        # Return the ELBO (a lower bound on the NLL)
        mu, sigma = self.encoder(x)

        kl_divergence = (mu**2 + sigma**2 - 1) / 2 - torch.log(sigma)
        kl_divergence = kl_divergence.sum(dim=1)

        latents = self.get_normal_sample(mu, sigma)

        reconstructions = self.decoder(latents)

        reconstruction_loss = (x - reconstructions) ** 2
        reconstruction_loss = reconstruction_loss.sum(dim=(1, 2, 3))

        ELBO = reconstruction_loss + kl_divergence

        return ELBO

    def generate_sample(self, batch_size):
        latents = self.N.sample((batch_size, self.latent_dims))
        samples = self.decoder(latents)

        return samples

    @staticmethod
    def load_serialised(model_name):

        save_path = path.join(VAE_ROOT, model_name + ".pt")
        checkpoint = torch.load(save_path)

        vae = SimpleVAE(encoder=LargeEncoder(), decoder=LargeDecoder()) # TODO: clean this
        vae.load_state_dict(checkpoint["vae_state_dict"])

        return vae


class LargeEncoder(nn.Module):
    def __init__(self, input_shape=(32, 32, 3), latent_dims=64):
        super().__init__()
        # Adapted from: https://github.com/realfolkcode/PyTorch-VAE-CIFAR10/blob/master/models/beta_vae.py

        input_width, input_height, input_channels = input_shape

        assert input_width == 32
        assert input_width == 32

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        in_channels = input_channels

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.conv_backbone = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dims)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dims)

    def forward(self, x):
        x = self.conv_backbone(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_var(x))
        return mu, sigma


class LargeDecoder(nn.Module):
    def __init__(self, input_shape=(32, 32, 3), latent_dims=64):
        super().__init__()
        # Adapted from: https://github.com/realfolkcode/PyTorch-VAE-CIFAR10/blob/master/models/beta_vae.py

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        hidden_dims.reverse()

        self.decoder_input = nn.Linear(latent_dims, 512)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                      kernel_size= 3, padding= 1),
            nn.Tanh()
        )

        self.conv_backbone = nn.Sequential(*modules)

    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, 512, 1, 1)
        x = self.conv_backbone(x)
        x = self.final_layer(x)
        x = x/2  # as the images are in the range (-0.5, 0.5)
        return x


class SmallEncoder(nn.Module):
    def __init__(self, input_shape=(32, 32, 3), latent_dims=10):
        super().__init__()

        input_width, input_height, input_channels = input_shape

        self.conv_1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=(2, 2))
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv_layers = [self.conv_1, self.conv_2, self.conv_3]
        self.intermediate_conv_size = 64 * input_width // 2 * input_width // 2

        self.linear_1 = nn.Linear(self.intermediate_conv_size, 32)
        self.mean_head = nn.Linear(32, latent_dims)
        self.std_head = nn.Linear(32, latent_dims)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = torch.relu(x)

        x = x.view(-1, self.intermediate_conv_size)
        x = self.linear_1(x)
        x = torch.relu(x)
        mu = self.mean_head(x)
        sigma = torch.exp(self.std_head(x))
        return mu, sigma


class SmallDecoder(nn.Module):
    def __init__(self, input_shape=(32, 32, 3), latent_dims=10):
        super().__init__()
        input_width, input_height, input_channels = input_shape

        self.inter_input_width = input_width // 2
        self.inter_input_height = input_height // 2

        self.linear_1 = nn.Linear(latent_dims, 32)
        self.linear_2 = nn.Linear(
            32, 32 * (self.inter_input_width) * (self.inter_input_height)
        )
        self.upsample = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=(2, 2))
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        x = torch.relu(x)

        x = x.view(-1, 32, self.inter_input_width, self.inter_input_height)
        x = self.upsample(x)

        x = torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = x - 0.5  # as the images are in the range (-0.5, 0.5)
        return x


def training_loop(
    n_epochs, optimizer, vae, train_loader, checkpoint_path, device, starting_epoch=1
):
    for n in range(starting_epoch, n_epochs + 1):
        train_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device=device)

            ELBOs = vae.eval_nll(imgs)
            loss = ELBOs.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if n < 5 or n % 10 == 0:
            print(f"epoch: {n}, train loss: {train_loss / len(train_loader)}")
            torch.save(
                {
                    "epoch": n,
                    "vae_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(device)

    for dataset_name in ["gtsrb"]:

        dataset_wrapper = to_dataset_wrapper[dataset_name]
        input_shape, _, train_dataset, _ = dataset_wrapper.get_all()

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )

        my_encoder = LargeEncoder(input_shape, latent_dims=64)
        my_decoder = LargeDecoder(input_shape, latent_dims=64)

        my_vae = SimpleVAE(input_shape=input_shape, encoder=my_encoder, decoder=my_decoder, latent_dims=64)

        pytorch_total_params = sum(p.numel() for p in my_vae.parameters())

        print("total parameters:", pytorch_total_params)

        my_optimizer = torch.optim.Adam(my_vae.parameters(), lr=3e-4)

        my_vae.to(device=device)

        print("training on", dataset_name)

        training_loop(
            n_epochs=100,
            optimizer=my_optimizer,
            vae=my_vae,
            train_loader=train_loader,
            checkpoint_path=path.join(VAE_ROOT, f"VAE_{dataset_name}.pt"),
            device=device,
        )
