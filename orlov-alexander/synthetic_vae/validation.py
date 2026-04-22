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

        orig_mean = orig["overall_mean"]
        orig_std = orig["overall_std"]
        orig_q25 = orig["overall_q25"]
        orig_q50 = orig["overall_q50"]
        orig_q75 = orig["overall_q75"]
        orig_iqr = orig["overall_iqr"]
        orig_skew_vec = orig["skew_vector"]
        orig_kurt_vec = orig["kurt_vector"]

        syn_mean_vec = np.mean(syn_vec, axis=0)
        syn_std_vec = np.std(syn_vec, axis=0)

        valid_mask = syn_std_vec > 1e-9
        collapsed_dims = int(np.sum(~valid_mask))
        final_mask = valid_mask & (~np.isnan(orig_skew_vec))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            syn_skew_vec = stats.skew(syn_vec, axis=0, nan_policy="omit")
            syn_kurt_vec = stats.kurtosis(syn_vec, axis=0, nan_policy="omit")

        skew_mae = float(np.mean(np.abs(orig_skew_vec[final_mask] - syn_skew_vec[final_mask]))) if final_mask.any() else float("nan")
        kurt_mae = float(np.mean(np.abs(orig_kurt_vec[final_mask] - syn_kurt_vec[final_mask]))) if final_mask.any() else float("nan")

        syn_mean = float(np.mean(syn_mean_vec))
        syn_std = float(np.mean(syn_std_vec))
        syn_q25 = float(np.mean(np.percentile(syn_vec, 25, axis=1)))
        syn_q50 = float(np.mean(np.percentile(syn_vec, 50, axis=1)))
        syn_q75 = float(np.mean(np.percentile(syn_vec, 75, axis=1)))
        syn_iqr = syn_q75 - syn_q25

        orig_vecs = original_embeddings[col]
        centr_cos = float(
            cosine_similarity(
                orig["mean_vector"].reshape(1, -1),
                syn_mean_vec.reshape(1, -1),
            )[0][0]
        )
        syn_div, _ = calculate_pairwise_diversity(syn_vec)
        orig_div, _ = calculate_pairwise_diversity(orig_vecs)

        def abs_err(s, o):
            return abs(s - o)

        def rel_err(s, o):
            return abs(s - o) / (abs(o) + 1e-8)

        rows.append(
            {
                "column": col,
                "n_original": orig["n_entries"],
                "n_synthetic": len(syn_vec),
                "vector_dim": syn_vec.shape[1],
                "collapsed_dims": collapsed_dims,
                "total_dims": syn_vec.shape[1],

                "orig_mean": orig_mean,
                "syn_mean": syn_mean,
                "orig_std": orig_std,
                "syn_std": syn_std,
                "orig_q25": orig_q25,
                "syn_q25": syn_q25,
                "orig_q50": orig_q50,
                "syn_q50": syn_q50,
                "orig_q75": orig_q75,
                "syn_q75": syn_q75,
                "orig_iqr": orig_iqr,
                "syn_iqr": syn_iqr,

                "centr_cos_sim": centr_cos,
                "syn_diversity_avg": syn_div,
                "orig_diversity_avg": orig_div,

                "abs_err_mean": abs_err(syn_mean, orig_mean),
                "abs_err_std": abs_err(syn_std, orig_std),
                "abs_err_q25": abs_err(syn_q25, orig_q25),
                "abs_err_q50": abs_err(syn_q50, orig_q50),
                "abs_err_q75": abs_err(syn_q75, orig_q75),
                "abs_err_iqr": abs_err(syn_iqr, orig_iqr),

                "rel_err_mean": rel_err(syn_mean, orig_mean),
                "rel_err_std": rel_err(syn_std, orig_std),
                "rel_err_q25": rel_err(syn_q25, orig_q25),
                "rel_err_q50": rel_err(syn_q50, orig_q50),
                "rel_err_q75": rel_err(syn_q75, orig_q75),
                "rel_err_iqr": rel_err(syn_iqr, orig_iqr),

                "skew_mae": skew_mae,
                "kurt_mae": kurt_mae,
            }
        )

    results_df = pd.DataFrame(rows)
    composite_cols = ["rel_err_mean", "rel_err_std", "rel_err_q25", "rel_err_q50", "rel_err_q75", "rel_err_iqr"]
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
        "orig_mean", "syn_mean", "abs_err_mean", "rel_err_mean",
        "orig_std", "syn_std", "abs_err_std", "rel_err_std",
        "orig_q25", "syn_q25", "abs_err_q25", "rel_err_q25",
        "orig_q50", "syn_q50", "abs_err_q50", "rel_err_q50",
        "orig_q75", "syn_q75", "abs_err_q75", "rel_err_q75",
        "orig_iqr", "syn_iqr", "abs_err_iqr", "rel_err_iqr",
        "composite_score", "centr_cos_sim",
        "syn_diversity_avg", "orig_diversity_avg",
        "skew_mae", "kurt_mae",
        "collapsed_dims", "total_dims",
    ]

    pivot = (
        detailed[["column"] + pivot_cols]
        .set_index("column")[pivot_cols]
        .T.round(6)
    )
    pivot.columns.name = "metric"
    return pivot, detailed
