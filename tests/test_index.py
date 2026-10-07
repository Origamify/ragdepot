"""Store tests: upsert semantics, deletes, stats, BM25."""

import numpy as np
import pytest

from ragdepot.depot import DepotError
from ragdepot.index import Store, content_hash, doc_id_for, fts_query


def vec(seed: float, dim: int = 4) -> bytes:
    v = np.full(dim, seed, dtype=np.float32)
    v /= np.linalg.norm(v)
    return v.tobytes()


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path, "testwiki")


def upsert_doc(store, text="alpha beta gamma", heading="H1", seed=0.5):
    return store.upsert(
        doc_id_for("testwiki", "doc1"),
        title="Doc One",
        path="docs/one.md",
        url="",
        text_hash=content_hash(text),
        chunks=[(heading, text, vec(seed))],
    )


def test_creates_db_in_scope_dir(tmp_path):
    store = Store(tmp_path, "mywiki")
    upsert_doc(store)
    assert (tmp_path / "mywiki" / "depot.db").is_file()


def test_bad_scope_name_rejected(tmp_path):
    with pytest.raises(DepotError):
        Store(tmp_path, "../escape")
    with pytest.raises(DepotError):
        Store(tmp_path, "")


def test_upsert_and_stats(store):
    assert upsert_doc(store) is True
    assert store.stats() == {"docs": 1, "chunks": 1}


def test_upsert_unchanged_is_noop(store):
    upsert_doc(store)
    assert upsert_doc(store) is False  # same content
    assert store.stats() == {"docs": 1, "chunks": 1}


def test_upsert_changed_replaces(store):
    upsert_doc(store, text="alpha beta")
    upsert_doc(store, text="totally different words", heading="H2")
    assert store.stats() == {"docs": 1, "chunks": 1}
    (row,) = store.fetch(list(store.vectors()[0])).values()
    assert row[5] == "H2" and "different" in row[6]


def test_multi_chunk_upsert_then_delete_doc(store):
    did = doc_id_for("testwiki", "doc1")
    store.upsert(
        did,
        title="T",
        path="",
        url="https://x",
        text_hash=content_hash("c"),
        chunks=[("h", f"text {i}", vec(0.1 + i * 0.1)) for i in range(3)],
    )
    assert store.stats() == {"docs": 1, "chunks": 3}
    assert store.delete_doc(did) == 1
    assert store.stats() == {"docs": 0, "chunks": 0}
    assert store.bm25("text", 10) == []


def test_delete_file(store):
    upsert_doc(store)
    assert store.delete_file("docs/one.md") == 1
    assert store.delete_file("docs/one.md") == 0
    assert store.stats() == {"docs": 0, "chunks": 0}


def test_bm25_ranks_exact_terms(store):
    store.upsert(
        doc_id_for("testwiki", "d1"),
        title="A",
        path="",
        url="",
        text_hash="h1",
        chunks=[("h", "solar panel output per rooftop", vec(0.1))],
    )
    store.upsert(
        doc_id_for("testwiki", "d2"),
        title="B",
        path="",
        url="",
        text_hash="h2",
        chunks=[("h", "hydro panel glossary terms with several more padding words", vec(0.2))],
    )
    store.upsert(
        doc_id_for("testwiki", "d3"),
        title="C",
        path="",
        url="",
        text_hash="h3",
        chunks=[("h", "garden furniture assembly guide", vec(0.3))],
    )
    # d4/d5 keep "panel" under half the corpus — FTS5 clamps IDF otherwise
    for i, filler in enumerate(["train timetable for the northern line", "roman pottery"]):
        store.upsert(
            doc_id_for("testwiki", f"d{4 + i}"),
            title=f"D{4 + i}",
            path="",
            url="",
            text_hash=f"h{4 + i}",
            chunks=[("h", filler, vec(0.4 + i * 0.1))],
        )
    hits = store.bm25("panel", 10)
    assert len(hits) == 2 and hits[0][0] == 1 and hits[0][1] < hits[1][1]


def test_bm25_special_chars_safe(store):
    upsert_doc(store)
    assert store.bm25('"; DROP TABLE chunks_fts; --', 5) == []


def test_vectors_shape(store):
    upsert_doc(store)
    ids, m = store.vectors()
    assert len(ids) == 1 and m.shape == (1, 4)
    np.testing.assert_allclose(np.linalg.norm(m[0]), 1.0, atol=1e-6)


def test_fts_query_ands_tokens():
    assert fts_query('solar "panel"! (output)') == '"solar" "panel" "output"'
    assert fts_query("!!!") == '""'
