"""
trainer.py — Обучение β-VAE по колонкам эмбеддингов.

Принимает features_data из stats.py, обучает отдельный VAE на каждой колонке
и возвращает синтетические векторы в том же формате (synthetic_data DataFrame).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from .model import VAE, LogCoshLoss, cyclical_beta, generate_vectors, vae_loss


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
    base_beta: float = 0.01,
    max_beta: float = 0.5,
    device: str = "cpu",
    verbose: bool = True,
) -> pd.DataFrame:
    norm_params = {
        item["column"]: {
            "mean_vec": item["mean_vector"],
            "std_vec": item["std_vector"],
        }
        for item in summary_stats
    }

    criterion = LogCoshLoss()
    results: list[dict] = []

    for item in features_data:
        column: str = item["column"]
        features: np.ndarray = item["features"]
        n_samples, input_dim = features.shape

        mean_vec = norm_params[column]["mean_vec"]
        std_vec = norm_params[column]["std_vec"]

        data_tensor = torch.FloatTensor(features).to(device)
        model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        total_loss = 0.0

        for epoch in range(epochs):
            model.train()
            beta = cyclical_beta(epoch, epochs, base_beta=base_beta, max_beta=max_beta)
            temp = 1.0 + 0.2 * np.sin(epoch / 10.0)

            perm = torch.randperm(n_samples, device=device)
            total_loss = 0.0

            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                batch = data_tensor[idx]

                optimizer.zero_grad()
                recon, mu, logvar = model(batch, temp=temp)
                loss = vae_loss(recon, batch, mu, logvar, beta=beta, criterion=criterion)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose:
                print(
                    f"  [{column}] epoch {epoch + 1:3d}/{epochs}"
                    f"  β={beta:.3f}  τ={temp:.3f}  loss={total_loss:.4f}"
                )

        syn_vec = generate_vectors(model, num_samples=num_synthetic, latent_dim=latent_dim, device=device)
        synthetic_vectors = syn_vec.cpu().numpy() * std_vec + mean_vec

        results.append(
            {
                "column": column,
                "n_entries": n_samples,
                "total_loss": total_loss,
                "synthetic_features": synthetic_vectors,
            }
        )

    return pd.DataFrame(results)
