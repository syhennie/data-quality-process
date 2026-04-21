"""
synthetic_vae — Пакет синтеза текстовых данных через β-VAE на sentence embeddings.

Модули:
  vectorizer   Загрузка SentenceTransformer и кодирование текста.
  stats        Вычисление описательных статистик эмбеддингов.
  model        Архитектура VAE, функции потерь, β-расписания.
  trainer      Цикл обучения VAE по колонкам, генерация синтетики.
  validation   Валидация (статистики, SWD, cosine, diversity).
  pipeline     Сквозной пайплайн SyntheticEmbeddingPipeline.
"""

from .model import VAE, LogCoshLoss, cyclical_beta, generate_vectors
from .pipeline import SyntheticEmbeddingPipeline
from .stats import compute_embedding_stats
from .trainer import train_vae_per_column
from .validation import (
    calculate_wasserstein_metrics,
    validate_synthetic_data,
    validate_synthetic_data_pivot,
)
from .vectorizer import load_model, text_to_embeddings

__all__ = [
    "SyntheticEmbeddingPipeline",
    "VAE",
    "LogCoshLoss",
    "cyclical_beta",
    "generate_vectors",
    "load_model",
    "text_to_embeddings",
    "compute_embedding_stats",
    "train_vae_per_column",
    "validate_synthetic_data",
    "validate_synthetic_data_pivot",
    "calculate_wasserstein_metrics",
]
