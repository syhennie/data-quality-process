"""
stats.py — Модуль для вычисления описательной статистики эмбеддингов.

Вычисляет per-dimension и scalar (overall) статистики по матрице эмбеддингов.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_embedding_stats(
    embeddings: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, list[dict]]:
    summary_rows: list[dict] = []
    features_data: list[dict] = []

    for column, emb in embeddings.items():
        n, dim = emb.shape

        mean_vec = np.mean(emb, axis=0)
        std_vec = np.std(emb, axis=0)
        skew_vec = stats.skew(emb, axis=0, nan_policy="omit")
        kurt_vec = stats.kurtosis(emb, axis=0, nan_policy="omit")
        q25_vec = np.percentile(emb, 25, axis=0)
        q50_vec = np.percentile(emb, 50, axis=0)
        q75_vec = np.percentile(emb, 75, axis=0)

        overall_mean = float(np.mean(mean_vec))
        overall_std = float(np.mean(std_vec))
        overall_q25 = float(np.mean(np.percentile(emb, 25, axis=1)))
        overall_q50 = float(np.mean(np.percentile(emb, 50, axis=1)))
        overall_q75 = float(np.mean(np.percentile(emb, 75, axis=1)))
        overall_iqr = overall_q75 - overall_q25

        summary_rows.append(
            {
                "column": column,
                "n_entries": n,
                "vector_dim": dim,
                "overall_mean": overall_mean,
                "overall_std": overall_std,
                "overall_q25": overall_q25,
                "overall_q50": overall_q50,
                "overall_q75": overall_q75,
                "overall_iqr": overall_iqr,
                "mean_vector": mean_vec,
                "std_vector": std_vec,
                "skew_vector": skew_vec,
                "kurt_vector": kurt_vec,
                "q25_vector": q25_vec,
                "q50_vector": q50_vec,
                "q75_vector": q75_vec,
            }
        )

        features_data.append(
            {
                "column": column,
                "features": emb,
                "raw_features": emb,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, features_data
