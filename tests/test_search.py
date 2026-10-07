"""Hybrid search tests: fusion, BM25 exact hits, scope merge."""

from conftest import fake_embed

from ragdepot.index import Store, content_hash, doc_id_for
from ragdepot.search import dense_top, rrf_fuse, search


def make_store(tmp_path, scope: str, docs: dict[str, str]) -> Store:
    store = Store(tmp_path, scope)
    for key, text in docs.items():
        store.upsert(
            doc_id_for(scope, key),
            title=key,
            path=f"{key}.md" if scope != "global" else "",
            url=f"https://wiki/{key}" if scope == "global" else "",
            text_hash=content_hash(text),
            chunks=[("heading of " + key, text, fake_embed([text])[0].tobytes())],
        )
    return store


def run_query(scopes, text, k=8):
    return search(scopes, text, fake_embed([text])[0], k)


def test_rrf_fuse_rewards_both_lists():
    only_dense = [10, 11, 12]
    only_bm25 = [20, 21, 22]
    both = [10, 20, 30]
    fused = rrf_fuse([only_dense, only_bm25, both])
    best = max(fused, key=fused.get)
    assert best in (10, 20)  # appears in two lists beats any single-list id
    assert fused[10] > fused[20] > fused[21]


def test_bm25_finds_exact_id_chunk(tmp_path):
    store = make_store(
        tmp_path,
        "local",
        {
            "errorlog": "error code X-991 raised by module gamma during startup",
            "cooking": "code of practice for kitchen hygiene and cooking safety",
        },
    )
    hits = run_query([("local", store)], "error code X-991")
    assert hits[0].snippet and "X-991" in hits[0].snippet


def test_fusion_ranks_both_list_hit_top(tmp_path):
    store = make_store(
        tmp_path,
        "local",
        {
            "energy": "solar panel output per rooftop detailed study",
            "hydro": "hydro panel production line packaging words filler text",
        },
    )
    hits = run_query([("local", store)], "solar panel output")
    assert "energy" in hits[0].title


def test_scope_merge_local_first_on_tie(tmp_path):
    text = "identical shared passage about quilted patchwork techniques"
    local = make_store(tmp_path, "mywiki", {"a": text})
    glob = make_store(tmp_path, "global", {"b": text})
    hits = run_query([("mywiki", local), ("global", glob)], "quilted patchwork techniques")
    assert len(hits) == 2
    assert hits[0].scope == "mywiki" and hits[0].score == hits[1].score


def test_hit_fields_and_snippet_heading(tmp_path):
    store = make_store(tmp_path, "local", {"doc": "alpha beta gamma delta words here"})
    hits = run_query([("local", store)], "alpha beta")
    h = hits[0]
    assert h.scope == "local" and h.doc_id and h.path == "doc.md" and h.url == ""
    assert h.snippet.startswith("[heading of doc]")


def test_empty_store_returns_no_hits(tmp_path):
    store = Store(tmp_path, "empty")
    assert run_query([("empty", store)], "anything") == []


def test_k_limits(tmp_path):
    store = make_store(
        tmp_path, "local", {f"d{i}": f"distinct topic number {i} words" for i in range(5)}
    )
    assert len(run_query([("local", store)], "distinct topic", k=2)) == 2


def test_dense_top_orders_by_similarity(tmp_path):
    near, far = "shared words appear here", "nothing in common at all"
    store = make_store(tmp_path, "local", {"near": near, "far": far})
    top = dense_top(store, fake_embed(["shared words"])[0], 2)
    assert top[0][1] >= top[1][1]
