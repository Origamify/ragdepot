"""ragdepot: standalone hybrid RAG library (dense + BM25 over sqlite/FTS5)."""

from ragdepot.depot import Depot, DepotError, Hit

__all__ = ["Depot", "DepotError", "Hit"]
