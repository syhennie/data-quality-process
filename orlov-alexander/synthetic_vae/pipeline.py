"""
pipeline.py — Сквозной пайплайн синтеза текстовых эмбеддингов через β-VAE.

Использование
-------------
from synthetic_vae.pipeline import SyntheticEmbeddingPipeline

pipe = SyntheticEmbeddingPipeline()
pipe.fit(df)

pivot, detailed, wasserstein = pipe.validate()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import compute_embedding_stats
from .trainer import train_vae_per_column
from .validation import (
    calculate_wasserstein_metrics,
    validate_synthetic_data_pivot,
)
from .vectorizer import load_model, text_to_embeddings


class SyntheticEmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        local_model_path: str = "./models/all-MiniLM-L6-v2",
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
    ) -> None:
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.num_synthetic = num_synthetic
        self.base_beta = base_beta
        self.max_beta = max_beta
        self.device = device
        self.verbose = verbose

        self.embeddings_: dict[str, np.ndarray] | None = None
        self.summary_df_: pd.DataFrame | None = None
        self.features_data_: list[dict] | None = None
        self.synthetic_data_: pd.DataFrame | None = None


    def fit(
        self,
        df: pd.DataFrame,
        text_cols: list[str] | None = None,
    ) -> "SyntheticEmbeddingPipeline":
        print("Загрузка embedding-модели")
        embed_model = load_model(self.model_name, self.local_model_path)
        print("Векторизация текста")
        self.embeddings_ = text_to_embeddings(df, embed_model, text_cols=text_cols)
        print("Вычисление статистик")
        self.summary_df_, self.features_data_ = compute_embedding_stats(self.embeddings_)
        print(self.summary_df_[["column", "n_entries", "vector_dim", "overall_mean", "overall_std"]].to_string(index=False))
        print("Обучение β-VAE и генерация синтетических векторов")
        self.synthetic_data_ = train_vae_per_column(
            self.features_data_,
            self.summary_df_.to_dict(orient="records"),
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
            num_synthetic=self.num_synthetic,
            base_beta=self.base_beta,
            max_beta=self.max_beta,
            device=self.device,
            verbose=self.verbose,
        )
        return self

    def validate(
        self,
        n_wasserstein_projections: int = 100,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self._check_fitted()

        pivot, detailed = validate_synthetic_data_pivot(
            self.synthetic_data_,
            self.summary_df_,
            self.features_data_,
        )

        wasserstein = calculate_wasserstein_metrics(
            original_embeddings={item["column"]: item["raw_features"] for item in self.features_data_},
            synthetic_data=self.synthetic_data_,
            n_projections=n_wasserstein_projections,
        )

        detailed = detailed.merge(wasserstein, on="column", how="left")
        if "normalized_swd" in detailed.columns:
            detailed["normalized_swd_capped"] = detailed["normalized_swd"].clip(upper=1.0)
            cols = ["rel_err_mean", "rel_err_std", "rel_err_q25", "rel_err_q50", "rel_err_q75", "rel_err_iqr", "normalized_swd_capped"]
            detailed["composite_score_with_swd"] = detailed[cols].mean(axis=1)

        return pivot, detailed, wasserstein


    def get_synthetic_embeddings(self) -> dict[str, np.ndarray]:
        self._check_fitted()
        return {
            row["column"]: np.asarray(row["synthetic_features"])
            for _, row in self.synthetic_data_.iterrows()
        }


    def _check_fitted(self) -> None:
        if self.synthetic_data_ is None:
            raise RuntimeError("Сначала вызовите .fit(df).")
