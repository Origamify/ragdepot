"""Depot facade tests: upsert/query/deletes/stats end-to-end
with the offline fake embedder, plus the degradation contract."""

import numpy as np
import pytest

from ragdepot import Depot, DepotError, Hit, embedder


@pytest.fixture
def depot(tmp_path, embed_fn):
    return Depot(root=tmp_path, embed_fn=embed_fn)


def write_md(tmp_path, name, text):
    p = tmp_path / f"{name}.md"
    p.write_text(text)
    return p


DOC = (
    "# Solar Panels\n\n"
    + "solar panel output data " * 30
    + "\n\n## Costs\n\n"
    + "installation cost analysis " * 30
)


def test_end_to_end_smoke(depot, tmp_path):
    f = write_md(tmp_path, "farm", DOC)
    depot.upsert_file(scope="mywiki", path=f)
    depot.upsert_text(
        scope="global",
        title="Energy overview",
        url="https://w/energy",
        text="home energy overview " * 40,
    )
    hits = depot.query("solar panel output", scope="mywiki", k=4)
    assert hits and all(isinstance(h, Hit) for h in hits)
    assert hits[0].scope == "mywiki" and hits[0].path == str(f)
    assert depot.stats() == {
        "mywiki": {"docs": 1, "chunks": 2},
        "global": {"docs": 1, "chunks": 1},
    }


def test_upsert_idempotent_and_replace(depot, tmp_path):
    f = write_md(tmp_path, "farm", DOC)
    assert depot.upsert_file("mywiki", f) is True
    assert depot.upsert_file("mywiki", f) is False  # unchanged
    f.write_text("# New Heading\n\n" + "completely different content " * 40)
    assert depot.upsert_file("mywiki", f) is True
    assert depot.stats()["mywiki"] == {"docs": 1, "chunks": 1}


def test_query_merges_scopes_local_first(depot, tmp_path):
    f = write_md(tmp_path, "farm", DOC)
    depot.upsert_file("mywiki", f)
    depot.upsert_text(
        scope="global",
        title="Solar",
        url="https://w/b",
        text="# Solar\n\n" + "solar panel output data " * 30,
    )
    hits = depot.query("solar panel output", scope=None)
    assert {h.scope for h in hits} == {"mywiki", "global"}


def test_headings_only_file_indexes_nothing(depot, tmp_path):
    f = write_md(tmp_path, "nav", "# Only\n\n## Headings\n")
    assert depot.upsert_file("mywiki", f) is False
    assert depot.stats() == {}


def test_upsert_clears_stale_doc_when_content_becomes_empty(depot, tmp_path):
    f = write_md(tmp_path, "farm", DOC)
    depot.upsert_file("mywiki", f)
    f.write_text("# Only\n\n## Headings\n")
    assert depot.upsert_file("mywiki", f) is True
    assert depot.stats() == {}


def test_path_identity_normalized(depot, tmp_path):
    f = write_md(tmp_path, "farm", DOC)
    dotted = tmp_path / "sub" / ".." / "farm.md"
    assert depot.upsert_file("mywiki", f) is True
    assert depot.upsert_file("mywiki", dotted) is False  # same resolved path
    assert depot.stats()["mywiki"] == {"docs": 1, "chunks": 2}
    assert depot.delete_file("mywiki", dotted) == 1


def test_delete_file_and_doc(depot, tmp_path):
    f = write_md(tmp_path, "farm", DOC)
    depot.upsert_file("mywiki", f)
    assert depot.delete_file("mywiki", f) == 1
    assert depot.stats() == {}
    depot.upsert_file("mywiki", f)
    for h in depot.query("solar", scope="mywiki"):
        depot.delete_doc(h.doc_id)
    assert depot.stats() == {}


def test_query_missing_scope_fails_cleanly(depot):
    with pytest.raises(DepotError, match="unavailable"):
        depot.query("anything", scope="nosuch")


def test_query_empty_root_returns_empty(depot):
    assert depot.query("anything", scope=None) == []
    assert depot.stats() == {}


def test_corrupt_store_fails_cleanly(tmp_path, embed_fn):
    (tmp_path / "broken" / "depot.db").parent.mkdir()
    (tmp_path / "broken" / "depot.db").write_bytes(b"not a sqlite file")
    depot = Depot(root=tmp_path, embed_fn=embed_fn)
    with pytest.raises(DepotError):
        depot.query("anything", scope="broken")


def test_upsert_missing_file_raises(depot):
    with pytest.raises(DepotError, match="cannot read"):
        depot.upsert_file("mywiki", "/nonexistent/file.md")


def test_upsert_non_utf8_file_raises_depot(depot, tmp_path):
    f = tmp_path / "bin.md"
    f.write_bytes(b"\xff\xfe\x00not utf8")
    with pytest.raises(DepotError, match="cannot read"):
        depot.upsert_file("mywiki", f)


def test_query_embedder_dim_mismatch_raises_depot(tmp_path, embed_fn):
    f = write_md(tmp_path, "farm", DOC)
    Depot(root=tmp_path, embed_fn=embed_fn).upsert_file("mywiki", f)
    other = Depot(root=tmp_path, embed_fn=lambda t: np.ones((len(t), 8), dtype=np.float32))
    with pytest.raises(DepotError, match="search failed"):
        other.query("solar", scope="mywiki")


def test_model_arg_reaches_embedder(tmp_path, monkeypatch):
    seen = {}

    def fake_embed(texts, model="unset"):
        seen[model] = len(texts)
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(embedder, "embed", fake_embed)
    f = write_md(tmp_path, "farm", DOC)
    Depot(root=tmp_path, model="some/model").upsert_file("mywiki", f)
    Depot(root=tmp_path).upsert_file("mywiki", f)  # default model
    assert set(seen) == {"some/model", embedder.MODEL_NAME}


def test_unknown_model_fails_as_depot_error(tmp_path):
    depot = Depot(root=tmp_path, model="no/such-model")
    with pytest.raises(DepotError, match="embedding failed"):
        depot.upsert_text("global", "t", "https://x", "words " * 30)


def test_construction_touches_no_disk(tmp_path):
    Depot(root=tmp_path / "does" / "not" / "exist")
    assert not (tmp_path / "does").exists()
