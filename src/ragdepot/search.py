"""Hybrid search: dense cosine top-N + FTS5 BM25 top-N,
fused via reciprocal rank fusion. scope=None merges scopes, local first."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragdepot.index import Store
from ragdepot.types import Hit

if TYPE_CHECKING:
    import numpy as np

RRF_K = 60  # standard reciprocal-rank-fusion constant
CANDIDATES = 32  # per-branch top-N fetched before fusion


def dense_top(store: Store, query_vec: np.ndarray, n: int) -> list[tuple[int, float]]:
    """Brute-force cosine top-n (chunk_id, similarity) over all vectors."""
    import numpy as np

    ids, matrix = store.vectors()
    if not ids:
        return []
    sims = matrix @ query_vec
    order = np.argsort(-sims)[:n]
    return [(ids[i], float(sims[i])) for i in order]


def rrf_fuse(rankings: list[list[int]]) -> dict[int, float]:
    """Reciprocal rank fusion: sum of 1/(RRF_K + rank) over each ranking."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return fused


def search(
    scopes: list[tuple[str, Store]],
    query_text: str,
    query_vec: np.ndarray,
    k: int,
) -> list[Hit]:
    """Hybrid search across (scope_name, store) pairs; local scopes first.

    `scopes` must already be ordered: local scopes before "global" —
    stable sorting keeps local hits above global on score ties.
    """
    n = max(CANDIDATES, k)
    merged: list[tuple[float, int, Hit]] = []  # (-score, insertion order, hit)
    seq = 0
    for scope, store in scopes:
        dense = dense_top(store, query_vec, n)
        lexical = store.bm25(query_text, n)
        fused = rrf_fuse([[cid for cid, _ in dense], [cid for cid, _ in lexical]])
        if not fused:
            continue
        rows = store.fetch(list(fused))
        for chunk_id, score in fused.items():
            row = rows.get(chunk_id)
            if row is None:
                continue
            _, doc_id, title, path, url, heading, text = row
            snippet = f"[{heading}] {text}" if heading else text
            merged.append((-score, seq, Hit(scope, doc_id, title, path, url, snippet, score)))
            seq += 1
    merged.sort()
    return [hit for _, _, hit in merged[:k]]
