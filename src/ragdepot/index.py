"""Storage + index: per-scope sqlite db at <root>/<scope>/depot.db.

WAL mode (concurrent readers, single writer); FTS5 table over chunk text for
BM25; vectors stored as float32 BLOBs, scored by brute-force cosine in numpy.
Upserts hash content and run in one transaction — a failure leaves the old
index intact. Connections are opened per operation and always closed.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from ragdepot.types import DepotError

if TYPE_CHECKING:
    from numpy.typing import NDArray

VALID_SCOPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")  # scope names are dir names

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id       TEXT PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT '',
    path         TEXT NOT NULL DEFAULT '',
    url          TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY,
    doc_id  TEXT NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
    heading TEXT NOT NULL DEFAULT '',
    text    TEXT NOT NULL,
    vector  BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='unicode61');
"""


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_id_for(scope: str, key: str) -> str:
    """Stable doc identity: scope + unique key (path, or url + title)."""
    return hashlib.sha256(f"{scope}\0{key}".encode()).hexdigest()


def fts_query(query: str) -> str:
    """Make a raw query safe for FTS5 MATCH: AND the quoted tokens."""
    tokens = re.findall(r"\w+", query, re.UNICODE)
    return " ".join(f'"{t}"' for t in tokens) if tokens else '""'


class Store:
    """One scope's index. All failures raise DepotError."""

    def __init__(self, root: Path, scope: str) -> None:
        if not VALID_SCOPE.match(scope):
            raise DepotError(f"invalid scope name: {scope!r}")
        self.db_path = root / scope / "depot.db"

    def _connect(self) -> sqlite3.Connection:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA)
            return conn
        except sqlite3.Error as e:
            raise DepotError(f"cannot open depot store {self.db_path}: {e}") from e

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        """One connection per operation: commit or rollback, then close."""
        conn = self._connect()
        try:
            with conn:  # transaction scope
                yield conn
        finally:
            conn.close()

    # -- writes -----------------------------------------------------------
    def upsert(
        self,
        doc_id: str,
        *,
        title: str,
        path: str,
        url: str,
        text_hash: str,
        chunks: list[tuple[str, str, bytes]],
    ) -> bool:
        """Replace a doc's chunks; (heading, text, vector-blob) per chunk.

        Unchanged content is a no-op returning False; a failed replace
        rolls back, leaving the old index intact.
        """
        try:
            with self._txn() as conn:
                row = conn.execute(
                    "SELECT content_hash FROM docs WHERE doc_id=?", (doc_id,)
                ).fetchone()
                if row and row[0] == text_hash:
                    return False
                self._delete_doc_rows(conn, "doc_id=?", (doc_id,))
                conn.execute(
                    "INSERT INTO docs(doc_id, title, path, url, content_hash) VALUES(?,?,?,?,?)",
                    (doc_id, title, path, url, text_hash),
                )
                for heading, ctext, vector in chunks:
                    cur = conn.execute(
                        "INSERT INTO chunks(doc_id, heading, text, vector) VALUES(?,?,?,?)",
                        (doc_id, heading, ctext, vector),
                    )
                    conn.execute(
                        "INSERT INTO chunks_fts(rowid, text) VALUES(?,?)",
                        (cur.lastrowid, ctext),
                    )
                return True
        except sqlite3.Error as e:
            raise DepotError(f"upsert failed in {self.db_path}: {e}") from e

    def delete_doc(self, doc_id: str) -> int:
        try:
            with self._txn() as conn:
                return self._delete_doc_rows(conn, "doc_id=?", (doc_id,))
        except sqlite3.Error as e:
            raise DepotError(f"delete_doc failed in {self.db_path}: {e}") from e

    def delete_file(self, path: str) -> int:
        try:
            with self._txn() as conn:
                return self._delete_doc_rows(conn, "path=?", (path,))
        except sqlite3.Error as e:
            raise DepotError(f"delete_file failed in {self.db_path}: {e}") from e

    @staticmethod
    def _delete_doc_rows(conn: sqlite3.Connection, where: str, args: tuple) -> int:
        conn.execute(
            f"DELETE FROM chunks_fts WHERE rowid IN (SELECT c.id FROM chunks c"  # noqa: S608
            f" JOIN docs d ON d.doc_id=c.doc_id WHERE d.{where})",
            args,
        )
        conn.execute(
            f"DELETE FROM chunks WHERE doc_id IN (SELECT doc_id FROM docs"  # noqa: S608
            f" WHERE {where})",
            args,
        )
        cur = conn.execute(f"DELETE FROM docs WHERE {where}", args)  # noqa: S608
        return cur.rowcount

    # -- reads ------------------------------------------------------------
    def vectors(self) -> tuple[list[int], NDArray]:
        """All chunk ids + vector matrix (n, dim) for brute-force cosine."""
        import numpy as np

        try:
            with self._txn() as conn:
                rows = conn.execute("SELECT id, vector FROM chunks").fetchall()
        except sqlite3.Error as e:
            raise DepotError(f"vector scan failed in {self.db_path}: {e}") from e
        ids = [r[0] for r in rows]
        if not rows:
            return ids, np.zeros((0, 1), dtype=np.float32)
        dim = len(rows[0][1]) // 4
        matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32)
        return ids, matrix.reshape(len(rows), dim)

    def bm25(self, query: str, n: int) -> list[tuple[int, float]]:
        """Top-n (chunk_id, bm25 score) from FTS5 (scores negative = better)."""
        try:
            with self._txn() as conn:
                return conn.execute(
                    "SELECT rowid, bm25(chunks_fts) FROM chunks_fts"
                    " WHERE chunks_fts MATCH ? ORDER BY 2 LIMIT ?",
                    (fts_query(query), n),
                ).fetchall()  # type: ignore[return-value]
        except sqlite3.Error as e:
            raise DepotError(f"bm25 failed in {self.db_path}: {e}") from e

    def fetch(self, chunk_ids: list[int]) -> dict[int, tuple]:
        """chunk rows (id, doc_id, title, path, url, heading, text) by id."""
        try:
            with self._txn() as conn:
                marks = ",".join("?" * len(chunk_ids))
                rows = conn.execute(
                    "SELECT c.id, d.doc_id, d.title, d.path, d.url, c.heading, c.text"
                    f" FROM chunks c JOIN docs d ON d.doc_id=c.doc_id"
                    f" WHERE c.id IN ({marks})",
                    chunk_ids,
                ).fetchall()
            return {r[0]: tuple(r) for r in rows}
        except sqlite3.Error as e:
            raise DepotError(f"fetch failed in {self.db_path}: {e}") from e

    def stats(self) -> dict[str, int]:
        try:
            with self._txn() as conn:
                docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
                chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            return {"docs": docs, "chunks": chunks}
        except sqlite3.Error as e:
            raise DepotError(f"stats failed in {self.db_path}: {e}") from e
