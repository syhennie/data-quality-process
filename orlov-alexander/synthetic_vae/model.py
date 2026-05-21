from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 50,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        temp: float = 1.0,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar) * temp
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        temp: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, temp=temp)
        return self.decode(z), mu, logvar


class LogCoshLoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_pred - y_true
        loss = torch.abs(diff) + F.softplus(-2.0 * torch.abs(diff)) - np.log(2.0)
        return torch.mean(loss)


def free_bits_kl_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.1,
) -> torch.Tensor:
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return torch.mean(kl_per_dim)


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.2,
    criterion: nn.Module | None = None,
) -> torch.Tensor:
    if criterion is None:
        criterion = LogCoshLoss()
    recon_loss = criterion(recon_x, x)
    kl = free_bits_kl_loss(mu, logvar)
    return recon_loss + beta * kl


def cyclical_beta(
    epoch: int,
    epochs: int,
    base_beta: float = 0.01,
    max_beta: float = 0.5,
) -> float:
    cycle = np.cos(2 * np.pi * epoch / epochs)
    return base_beta + 0.5 * (max_beta - base_beta) * (cycle + 1)


def linear_warmup_beta(
    epoch: int,
    warmup_epochs: int = 50,
    max_beta: float = 0.4,
    total_epochs=100,
) -> float:
    decay = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return max_beta * (1 - 0.5 * decay)


def generate_vectors(
    model: VAE,
    num_samples: int = 1000,
    latent_dim: int = 50,
    device: str = "cpu",
) -> torch.Tensor:
    temperature = 1.2
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device) * temperature
        return model.decode(z)
