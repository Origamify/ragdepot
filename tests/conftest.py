"""Shared test fixtures: deterministic fake embedder (word-hash vectors)."""

import re
import zlib

import numpy as np
import pytest

DIM = 64


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-words vectors; enough signal for fusion tests."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            out[i, zlib.crc32(word.encode()) % DIM] += 1.0
    norms = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out / norms


@pytest.fixture
def embed_fn():
    return fake_embed
