"""Embedder: local fastembed models, lazy load on first use,
returns row-normalized float32 vectors. No network after first model download.

One model per depot root: stored vectors are only comparable if every doc
and query used the same model. Switching models requires re-indexing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

MODEL_NAME = "minishlab/potion-base-8M"  # default: 256-dim static embeddings, tiny + fast

_models: dict[str, object] = {}  # lazy per-model singletons


def _get_model(model_name: str):
    if model_name not in _models:
        from fastembed import TextEmbedding

        _models[model_name] = TextEmbedding(model_name=model_name)
    return _models[model_name]


def embed(texts: list[str], model: str = MODEL_NAME) -> np.ndarray:
    """Embed texts -> (n, dim) float32 with unit-length rows.

    `model`: any fastembed-supported name (TextEmbedding.list_supported_models()).
    """
    import numpy as np

    if not texts:
        return np.zeros((0, 1), dtype=np.float32)  # no rows: dim is moot
    vectors = np.stack(list(_get_model(model).embed(texts))).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
