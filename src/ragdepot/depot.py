"""Depot facade — the public Python interface (rag-spec.md §Python interface).

Graceful degradation contract: Depot(root) never touches disk; all
client-facing failures raise DepotError so callers can catch and degrade
(e.g. answer "knowledge base unavailable").
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from ragdepot import embedder
from ragdepot.chunker import chunk
from ragdepot.index import Store, content_hash, doc_id_for
from ragdepot.search import search
from ragdepot.types import DepotError, Hit

__all__ = ["Depot", "DepotError", "Hit"]


class Depot:
    """Entry point. Construction is pure; a broken root surfaces on first use.

    `model`: any fastembed-supported model name (default potion-base-8M).
    One model per depot root — switching models requires re-indexing.
    `embed_fn` overrides both (tests inject a deterministic fake).
    """

    def __init__(self, root: str | Path, embed_fn=None, model: str | None = None) -> None:
        self.root = Path(root)
        self._embed = embed_fn or partial(embedder.embed, model=model or embedder.MODEL_NAME)

    # -- scope helpers ---------------------------------------------------
    def _store(self, scope: str) -> Store:
        try:
            return Store(self.root, scope)
        except DepotError:
            raise
        except Exception as e:  # defensive: scope handling must not leak types
            raise DepotError(f"bad scope {scope!r}: {e}") from e

    def _existing_scopes(self) -> list[str]:
        """Scope dirs that actually have an index; locals first, global last."""
        if not self.root.is_dir():
            return []
        names = [d.name for d in self.root.iterdir() if (d / "depot.db").is_file()]
        return [n for n in sorted(names) if n != "global"] + (
            ["global"] if "global" in names else []
        )

    def _scopes_for_query(self, scope: str | None) -> list[tuple[str, Store]]:
        if scope is not None:
            if not (self.root / scope / "depot.db").is_file():
                raise DepotError(f"knowledge base unavailable: no index for scope {scope!r}")
            return [(scope, self._store(scope))]
        return [(name, self._store(name)) for name in self._existing_scopes()]

    # -- indexing ---------------------------------------------------------
    def upsert_file(self, scope: str, path: str | Path) -> bool:
        """Index a local file. Idempotent; changed content replaces old chunks.

        Identity is the resolved absolute path ("a.md" and "./a.md" are one doc).
        """
        path = Path(path).resolve()
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError) as e:
            raise DepotError(f"cannot read {path}: {e}") from e
        return self._upsert(
            self._store(scope),
            doc_id_for(scope, str(path)),
            title=path.stem,
            path=str(path),
            url="",
            text=text,
        )

    def upsert_text(self, scope: str, title: str, url: str, text: str) -> bool:
        """Index title+url text into a scope (global corpus entries)."""
        return self._upsert(
            self._store(scope),
            doc_id_for(scope, f"{url}\0{title}"),
            title=title,
            path="",
            url=url,
            text=text,
        )

    def _upsert(
        self, store: Store, doc_id: str, *, title: str, path: str, url: str, text: str
    ) -> bool:
        chunks = chunk(text)
        if not chunks:  # nothing indexable (e.g. headings-only); drop any stale doc
            return store.delete_doc(doc_id) > 0
        try:
            vectors = self._embed([c.text for c in chunks])
        except Exception as e:
            raise DepotError(f"embedding failed: {e}") from e
        rows = [(c.heading, c.text, vectors[i].tobytes()) for i, c in enumerate(chunks)]
        return store.upsert(
            doc_id,
            title=title,
            path=path,
            url=url,
            text_hash=content_hash(text),
            chunks=rows,
        )

    # -- retrieval --------------------------------------------------------
    def query(self, text: str, scope: str | None = None, k: int = 8) -> list[Hit]:
        """Hybrid dense+BM25 search; scope=None merges scopes, local first."""
        try:
            query_vec = self._embed([text])[0]
        except Exception as e:
            raise DepotError(f"embedding failed: {e}") from e
        try:
            return search(self._scopes_for_query(scope), text, query_vec, k)
        except DepotError:
            raise
        except Exception as e:
            raise DepotError(f"search failed: {e}") from e

    # -- maintenance --------------------------------------------------------
    def delete_doc(self, doc_id: str) -> None:
        for name in self._existing_scopes():
            self._store(name).delete_doc(doc_id)

    def delete_file(self, scope: str, path: str | Path) -> int:
        return self._store(scope).delete_file(str(Path(path).resolve()))

    def stats(self) -> dict[str, dict[str, int]]:
        """Per-scope doc/chunk counts; scopes with no docs are omitted."""
        return {
            name: s
            for name in self._existing_scopes()
            if (s := self._store(name).stats())["docs"] > 0
        }
