from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity


def sliced_wasserstein_distance(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 100,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    swd = 0.0
    for _ in range(n_projections):
        theta = rng.standard_normal(d)
        theta /= np.linalg.norm(theta)
        swd += wasserstein_distance(X @ theta, Y @ theta)
    return swd / n_projections


def calculate_pairwise_diversity(
    embeddings: np.ndarray,
    sample_size: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(embeddings) > sample_size:
        idx = rng.choice(len(embeddings), sample_size, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings

    sim = cosine_similarity(sample)
    upper = np.triu_indices(sim.shape[0], k=1)
    vals = sim[upper]
    return float(np.mean(vals)), float(np.std(vals))


def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _per_dim_mae(orig_vec: np.ndarray, syn_vec: np.ndarray) -> float:
    return float(np.mean(np.abs(orig_vec - syn_vec)))


def _per_dim_rel_mae(orig_vec: np.ndarray, syn_vec: np.ndarray) -> float:
    return float(np.mean(np.abs(orig_vec - syn_vec) / (np.abs(orig_vec) + 1e-10)))


def validate_synthetic_data(
    synthetic_data: pd.DataFrame,
    summary_df: pd.DataFrame,
    features_data: list[dict],
) -> pd.DataFrame:
    summary_dict = summary_df.set_index("column").to_dict(orient="index")
    original_embeddings = {item["column"]: _to_numpy(item["raw_features"]) for item in features_data}

    rows: list[dict] = []

    for _, row in synthetic_data.iterrows():
        col: str = row["column"]
        syn_vec = _to_numpy(row["synthetic_features"])

        orig = summary_dict[col]

        orig_mean_vec = orig["mean_vector"]
        orig_std_vec = orig["std_vector"]
        orig_q25_vec = orig["q25_vector"]
        orig_q50_vec = orig["q50_vector"]
        orig_q75_vec = orig["q75_vector"]
        orig_iqr_vec = orig["iqr_vector"]
        orig_skew_vec = orig["skew_vector"]
        orig_kurt_vec = orig["kurt_vector"]

        syn_mean_vec = np.mean(syn_vec, axis=0)
        syn_std_vec = np.std(syn_vec, axis=0)
        syn_q25_vec = np.percentile(syn_vec, 25, axis=0)
        syn_q50_vec = np.percentile(syn_vec, 50, axis=0)
        syn_q75_vec = np.percentile(syn_vec, 75, axis=0)
        syn_iqr_vec = syn_q75_vec - syn_q25_vec

        valid_mask = syn_std_vec > 1e-9
        collapsed_dims = int(np.sum(~valid_mask))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            syn_skew_vec = stats.skew(syn_vec, axis=0, nan_policy="omit")
            syn_kurt_vec = stats.kurtosis(syn_vec, axis=0, nan_policy="omit")

        mean_mae = _per_dim_mae(orig_mean_vec, syn_mean_vec)
        std_mae = _per_dim_mae(orig_std_vec, syn_std_vec)
        q25_mae = _per_dim_mae(orig_q25_vec, syn_q25_vec)
        q50_mae = _per_dim_mae(orig_q50_vec, syn_q50_vec)
        q75_mae = _per_dim_mae(orig_q75_vec, syn_q75_vec)
        iqr_mae = _per_dim_mae(orig_iqr_vec, syn_iqr_vec)
        skew_mae = _per_dim_mae(orig_skew_vec, syn_skew_vec)
        kurt_mae = _per_dim_mae(orig_kurt_vec, syn_kurt_vec)

        mean_rel_mae = _per_dim_rel_mae(orig_mean_vec, syn_mean_vec)
        std_rel_mae = _per_dim_rel_mae(orig_std_vec, syn_std_vec)
        q25_rel_mae = _per_dim_rel_mae(orig_q25_vec, syn_q25_vec)
        q50_rel_mae = _per_dim_rel_mae(orig_q50_vec, syn_q50_vec)
        q75_rel_mae = _per_dim_rel_mae(orig_q75_vec, syn_q75_vec)
        iqr_rel_mae = _per_dim_rel_mae(orig_iqr_vec, syn_iqr_vec)

        orig_vecs = original_embeddings[col]
        centr_cos = float(
            cosine_similarity(
                orig["mean_vector"].reshape(1, -1),
                syn_mean_vec.reshape(1, -1),
            )[0][0]
        )
        syn_div, _ = calculate_pairwise_diversity(syn_vec)
        orig_div, _ = calculate_pairwise_diversity(orig_vecs)

        rows.append(
            {
                "column": col,
                "n_original": orig["n_entries"],
                "n_synthetic": len(syn_vec),
                "vector_dim": syn_vec.shape[1],
                "collapsed_dims": collapsed_dims,
                "total_dims": syn_vec.shape[1],

                "mean_mae":      mean_mae,
                "std_mae":       std_mae,
                "q25_mae":       q25_mae,
                "q50_mae":       q50_mae,
                "q75_mae":       q75_mae,
                "iqr_mae":       iqr_mae,
                "skew_mae":      skew_mae,
                "kurt_mae":      kurt_mae,

                "mean_rel_mae": mean_rel_mae,
                "std_rel_mae": std_rel_mae,
                "q25_rel_mae": q25_rel_mae,
                "q50_rel_mae": q50_rel_mae,
                "q75_rel_mae": q75_rel_mae,
                "iqr_rel_mae": iqr_rel_mae,

                "centr_cos_sim": centr_cos,
                "syn_diversity_avg": syn_div,
                "orig_diversity_avg": orig_div,

                "_orig_mean_vec": orig_mean_vec,
                "_syn_mean_vec": syn_mean_vec,
                "_orig_std_vec": orig_std_vec,
                "_syn_std_vec": syn_std_vec,
                "_orig_q25_vec": orig_q25_vec,
                "_syn_q25_vec": syn_q25_vec,
                "_orig_q50_vec": orig_q50_vec,
                "_syn_q50_vec": syn_q50_vec,
                "_orig_q75_vec": orig_q75_vec,
                "_syn_q75_vec": syn_q75_vec,
                "_orig_iqr_vec": orig_iqr_vec,
                "_syn_iqr_vec": syn_iqr_vec,
            }
        )

    results_df = pd.DataFrame(rows)
    composite_cols = [
        "mean_rel_mae", "std_rel_mae",
        "q25_rel_mae", "q50_rel_mae", "q75_rel_mae", "iqr_rel_mae",
    ]
    results_df["composite_score"] = results_df[composite_cols].mean(axis=1)
    return results_df


def calculate_wasserstein_metrics(
    original_embeddings: dict[str, np.ndarray],
    synthetic_data: pd.DataFrame,
    n_projections: int = 100,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in synthetic_data.iterrows():
        col: str = row["column"]
        syn_vec = _to_numpy(row["synthetic_features"])

        if col not in original_embeddings:
            print(f"  warning: нет оригинальных эмбеддингов для '{col}'")
            continue

        orig_vec = original_embeddings[col]
        swd = sliced_wasserstein_distance(orig_vec, syn_vec, n_projections=n_projections)
        sample_size = min(1000, len(orig_vec))
        idx = np.random.choice(len(orig_vec), sample_size, replace=False)
        baseline = float(np.mean(pdist(orig_vec[idx], metric="euclidean")))
        normalized = swd / baseline if baseline > 0 else swd

        rows.append(
            {
                "column": col,
                "sliced_wasserstein": swd,
                "baseline_distance": baseline,
                "normalized_swd": normalized,
            }
        )

    return pd.DataFrame(rows)


def validate_synthetic_data_pivot(
    synthetic_data: pd.DataFrame,
    summary_df: pd.DataFrame,
    features_data: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed = validate_synthetic_data(synthetic_data, summary_df, features_data)

    pivot_cols = [
        "mean_mae", "std_mae", "q25_mae", "q50_mae",
        "q75_mae", "iqr_mae", "skew_mae", "kurt_mae",
        "mean_rel_mae", "std_rel_mae", "q25_rel_mae", "q50_rel_mae",
        "q75_rel_mae", "iqr_rel_mae",
        "composite_score",
        "centr_cos_sim",
        "syn_diversity_avg", "orig_diversity_avg",
        "collapsed_dims",
    ]

    pivot = (
        detailed[["column"] + pivot_cols]
        .set_index("column")[pivot_cols]
        .T.round(6)
    )
    pivot.columns.name = "metric"
    return pivot, detailed
