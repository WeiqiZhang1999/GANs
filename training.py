#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/31/2022 4:57 PM
# @Author  : ZHANG WEIQI
# @File    : training.py
# @Software: PyCharm
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# write classes of Discriminator and Generator.
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),  # LeakyReLU formula: y = max(0, x) + leak * min(0, x)
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


# Hyper parameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of batches")
parser.add_argument("--z_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--lr", type=float, default=3e-4, help="adam: learning rate")
parser.add_argument("--image_size", type=int, default=28, help="size of each image")
parser.add_argument("--image_channels", type=int, default=1, help="channel(s) of each image")
opt = parser.parse_args()
image_dim = opt.image_size * opt.image_size * opt.image_channels
lr = opt.lr
z_dim = opt.z_dim
batch_size = opt.batch_size
num_epochs = opt.num_epochs
image_channels = opt.image_channels
print(opt)

# create models
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

# set random noise
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# set transforms
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize(),
        transforms.Normalize([0.5 for _ in range(image_channels)], [0.5 for _ in range(image_channels)])
    ]
)

# prepare datasets
dataset = datasets.MNIST(root="./dataset/", train=True, transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# prepare optimizers
optimizer_disc = optim.Adam(disc.parameters(), lr=lr)
optimizer_gen = optim.Adam(gen.parameters(), lr=lr)

# criterion
criterion = nn.BCELoss()

# tensorboard
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

# training loop
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### training discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_disc.step()

        ### training generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward(retain_graph=True)
        optimizer_gen.step()

        if batch_idx == 0:
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Image", img_grid_fake, global_step=step
                )
                writer_fake.add_image(
                    "Mnist Real Image", img_grid_real, global_step=step
                )
