"""Embedder tests: unit behavior plus one real-model test (the model is
downloaded to the user cache on first run)."""

import numpy as np

from ragdepot import embedder


def test_embed_deterministic_and_unit_length():
    v = embedder.embed(["hello world", "solar panel output"])
    assert v.dtype == np.float32 and v.shape[0] == 2 and v.shape[1] >= 256
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    v2 = embedder.embed(["hello world"])
    np.testing.assert_array_equal(v[0], v2[0])


def test_embed_empty():
    assert embedder.embed([]).shape == (0, 1)


def test_model_loaded_once_per_name(monkeypatch):
    import fastembed

    created = []

    class FakeTE:
        def __init__(self, model_name):
            created.append(model_name)

        def embed(self, texts):
            return [np.zeros(3, dtype=np.float32) for _ in texts]

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTE)
    embedder._models.clear()
    try:
        embedder.embed(["a"], model="m1")
        embedder.embed(["b"], model="m1")
        embedder.embed(["c"], model="m2")
    finally:
        embedder._models.clear()
    assert created == ["m1", "m2"]
