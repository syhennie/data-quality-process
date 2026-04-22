from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from .model import VAE, LogCoshLoss, generate_vectors, vae_loss


def _warmup_beta(epoch: int, epochs: int, max_beta: float, warmup_ratio: float = 0.3) -> float:
    warmup_epochs = max(1, int(epochs * warmup_ratio))
    if epoch < warmup_epochs:
        return max_beta * (epoch / warmup_epochs)
    return max_beta


def train_vae_per_column(
    features_data: list[dict],
    summary_stats: list[dict],
    *,
    hidden_dim: int = 128,
    latent_dim: int = 50,
    batch_size: int = 128,
    epochs: int = 100,
    lr: float = 1e-4,
    num_synthetic: int = 1000,
    base_beta: float = 0.01,   #
    max_beta: float = 0.05,
    warmup_ratio: float = 0.3,
    device: str = "cpu",
    verbose: bool = True,
) -> pd.DataFrame:
    norm_params = {
        item["column"]: {
            "mean_vec": item["mean_vec"],
            "std_vec":  item["std_vec"],
        }
        for item in features_data
    }

    criterion = LogCoshLoss()
    results: list[dict] = []

    for item in features_data:
        column: str = item["column"]
        features: np.ndarray = item["features"]
        n_samples, input_dim = features.shape

        mean_vec = norm_params[column]["mean_vec"]
        std_vec  = norm_params[column]["std_vec"]

        data_tensor = torch.FloatTensor(features).to(device)
        model       = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
        optimizer   = optim.Adam(model.parameters(), lr=lr)

        loss_history: list[float] = []
        total_loss = 0.0

        for epoch in range(epochs):
            model.train()
            beta       = _warmup_beta(epoch, epochs, max_beta, warmup_ratio)
            total_loss = 0.0
            perm       = torch.randperm(n_samples, device=device)

            for i in range(0, n_samples, batch_size):
                idx   = perm[i : i + batch_size]
                batch = data_tensor[idx]

                optimizer.zero_grad()
                recon, mu, logvar = model(batch)          # temp=1.0 locked
                loss = vae_loss(recon, batch, mu, logvar, beta=beta, criterion=criterion)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            loss_history.append(total_loss)

            if verbose:
                print(
                    f"  [{column}] epoch {epoch + 1:3d}/{epochs}"
                    f"  β={beta:.4f}  loss={total_loss:.4f}"
                )

        syn_vec          = generate_vectors(model, num_samples=num_synthetic, latent_dim=latent_dim, device=device)
        synthetic_vectors = syn_vec.cpu().numpy() * std_vec + mean_vec

        results.append(
            {
                "column":            column,
                "n_entries":         n_samples,
                "total_loss":        total_loss,
                "synthetic_features": synthetic_vectors,
                "loss_history":      loss_history,
            }
        )

    return pd.DataFrame(results)
