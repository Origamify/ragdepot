"""Shared types: the single client-facing error and the search-hit shape.

Leaf module (no ragdepot imports) so index/search/depot can all use them
without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass


class DepotError(Exception):
    """Single client-facing failure type ("knowledge base unavailable")."""


@dataclass(frozen=True)
class Hit:
    """One search result; snippet is the chunk text (prefixed by heading)."""

    scope: str
    doc_id: str
    title: str
    path: str
    url: str
    snippet: str
    score: float
