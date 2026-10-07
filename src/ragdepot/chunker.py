"""Chunker: split on blank-line paragraphs, pack into ~800-word
chunks, attach nearest preceding heading, drop short fragments. Pure, no I/O."""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_WORDS = 800  # max words per packed chunk
MIN_WORDS = 20  # chunks below this are navigation junk, dropped

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Chunk:
    """One indexed unit; heading is the nearest preceding ATX heading."""

    text: str
    heading: str


def _blocks(text: str) -> list[str]:
    """Split into non-empty, whitespace-stripped blank-line-separated blocks."""
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def _is_heading(block: str) -> bool:
    return bool(_ATX_HEADING.match(block)) and "\n" not in block


def _words(block: str) -> int:
    return len(block.split())


def chunk(text: str) -> list[Chunk]:
    """Pack paragraphs into ~TARGET_WORDS chunks, tagged with their heading.

    A new chunk starts when the word budget would overflow or when the
    nearest preceding heading changes. An oversized single paragraph
    becomes one chunk (paragraphs are never split). Heading-only input
    and fragments under MIN_WORDS are dropped.
    """
    chunks: list[Chunk] = []
    current_heading = ""
    buf: list[str] = []
    buf_words = 0

    def flush() -> None:
        nonlocal buf, buf_words
        if buf and buf_words >= MIN_WORDS:
            chunks.append(Chunk(text="\n\n".join(buf), heading=current_heading))
        buf, buf_words = [], 0

    for block in _blocks(text):
        if _is_heading(block):
            flush()  # never merge paragraphs across a heading change
            current_heading = _ATX_HEADING.match(block).group(2).strip()  # type: ignore[union-attr]
            continue
        if buf and buf_words + _words(block) > TARGET_WORDS:
            flush()
        buf.append(block)
        buf_words += _words(block)

    flush()
    return chunks
