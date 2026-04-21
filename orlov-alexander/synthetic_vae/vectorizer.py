"""
vectorizer.py — Модуль для получения эмбеддингов текста.

Поддерживает SentenceTransformer (all-MiniLM-L6-v2 по умолчанию).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LOCAL_PATH = "./models/all-MiniLM-L6-v2"


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    local_path: str = DEFAULT_LOCAL_PATH,
    save_local: bool = True,
) -> SentenceTransformer:
    local = Path(local_path)
    if local.exists():
        print(f"Загружаем модель из локального кэша: {local_path}")
        return SentenceTransformer(local_path)

    print(f"Загружаем модель с HuggingFace: {model_name}")
    model = SentenceTransformer(model_name)
    if save_local:
        model.save(local_path)
        print(f"Модель сохранена в {local_path}")
    return model


def text_to_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    text_cols: list[str] | None = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    if text_cols is None:
        text_cols = df.select_dtypes(include="object").columns.tolist()

    embeddings: dict[str, np.ndarray] = {}
    for col in text_cols:
        print(f"  Кодируем колонку '{col}'...")
        texts = df[col].fillna(" ").astype(str).tolist()
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        print(f"  → shape: {emb.shape}")
        embeddings[col] = emb

    return embeddings
