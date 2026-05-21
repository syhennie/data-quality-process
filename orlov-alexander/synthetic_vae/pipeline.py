from __future__ import annotations

import numpy as np
import pandas as pd

from .decoder import EmbeddingToTextDecoder
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
        max_beta: float = 0.05,
        warmup_ratio: float = 0.3,
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
        self.warmup_ratio = warmup_ratio
        self.device = device
        self.verbose = verbose

        self.embeddings_: dict[str, np.ndarray] | None = None
        self.summary_df_: pd.DataFrame | None = None
        self.features_data_: list[dict] | None = None
        self.synthetic_data_: pd.DataFrame | None = None

        self.decoders_: dict[str, EmbeddingToTextDecoder] = {}
        self._source_texts_: dict[str, list[str]] = {}

    def fit(
        self,
        df: pd.DataFrame,
        text_cols: list[str] | None = None,
    ) -> "SyntheticEmbeddingPipeline":
        embed_model = load_model(self.model_name, self.local_model_path)
        self.embeddings_ = text_to_embeddings(df, embed_model, text_cols=text_cols)
        self.summary_df_, self.features_data_ = compute_embedding_stats(self.embeddings_)
        print(self.summary_df_[["column", "n_entries", "vector_dim", "overall_mean", "overall_std"]].to_string(index=False))

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
            warmup_ratio=self.warmup_ratio,
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
            cols = ["mean_rel_mae", "std_rel_mae", "q25_rel_mae", "q50_rel_mae", "q75_rel_mae", "iqr_rel_mae", "normalized_swd_capped"]
            detailed["composite_score_with_swd"] = detailed[cols].mean(axis=1)

        return pivot, detailed, wasserstein

    def get_synthetic_embeddings(self) -> dict[str, np.ndarray]:
        self._check_fitted()
        return {
            row["column"]: np.asarray(row["synthetic_features"])
            for _, row in self.synthetic_data_.iterrows()
        }

    def fit_decoder(
        self,
        df: pd.DataFrame,
        text_cols: list[str] | None = None,
        decoder_model_name: str = "google/t5-efficient-tiny",
        decoder_epochs: int | None = None,
        decoder_batch_size: int = 32,
        decoder_lr: float = 3e-4,
    ) -> "SyntheticEmbeddingPipeline":
        self._check_fitted()

        if text_cols is None:
            text_cols = df.select_dtypes(include="object").columns.tolist()

        for col in text_cols:
            if col not in self.embeddings_:
                print(f" {col!r} не найдена в эмбеддингах, пропускаем.")
                continue

            texts = df[col].fillna(" ").astype(str).tolist()
            raw = next(f for f in self.features_data_ if f["column"] == col)
            embeddings = raw["raw_features"]
            print(f"\n  col: {col!r}  ({len(texts)} текстов, {embeddings.shape[1]}d)")

            decoder = EmbeddingToTextDecoder(
                t5_model_name=decoder_model_name,
                device=self.device,
                batch_size=decoder_batch_size,
                lr=decoder_lr,
                verbose=self.verbose,
            )
            decoder.fit(embeddings, texts, epochs=decoder_epochs)

            self.decoders_[col] = decoder
            self._source_texts_[col] = texts

        return self

    def decode_texts(
        self,
        col: str,
        synthetic_embeddings: np.ndarray | None = None,
        n: int | None = None,
        **decode_kwargs,
    ) -> list[str]:
        if col not in self.decoders_:
            raise RuntimeError(
                f"Декодер для {col!r} не обучен. "
                f"Вызов .fit_decoder() сначала."
            )

        if synthetic_embeddings is None:
            self._check_fitted()
            row = self.synthetic_data_[self.synthetic_data_["column"] == col]
            if row.empty:
                raise RuntimeError(f"Синтетические эмбеддинги для {col!r} не найдены.")
            synthetic_embeddings = np.asarray(row.iloc[0]["synthetic_features"])

        if n is not None:
            synthetic_embeddings = synthetic_embeddings[:n]

        return self.decoders_[col].decode(synthetic_embeddings, **decode_kwargs)

    def _check_fitted(self) -> None:
        if self.synthetic_data_ is None:
            raise RuntimeError("Сначала вызов .fit(df).")
