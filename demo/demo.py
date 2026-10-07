"""End-to-end ragdepot demo: index a Wikipedia article, then search it.

This is the whole library in four calls:

    depot = Depot(root=...)                      # 1. create the depot (pure, no disk I/O)
    depot.upsert_text(scope=..., ...)            # 2. store a document (chunk -> embed -> index)
    hits = depot.query("some question")          # 3. hybrid search (dense + BM25, RRF-fused)
    depot.stats()                                # 4. what's in there?

Usage:
    python demo/demo.py    (from the repo root, in a venv with
    ragdepot + fastembed installed)

What happens on each run:
    - fetches the plain text of the Wikipedia article "Roman Empire"
      (the only network call besides a one-time model download);
    - chunks it (~800-word pieces, each tagged with its nearest heading),
      embeds every chunk, and stores everything in a local sqlite index;
    - runs a few fact queries and prints the top hits.

Notes:
    - The embedder model (~30 MB) is downloaded once into the user cache;
      after that the library is fully offline.
    - Repeat runs are idempotent: the content hash matches, so the upsert
      is a no-op ("unchanged") and the index is left as-is.
    - The depot lives in demo/depot-data/ (gitignored) — delete that
      folder to start fresh.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

from ragdepot import Depot, DepotError

# -- configuration ---------------------------------------------------------

TITLE = "Roman_Empire"  # Wikipedia page to index (URL slug form)

# MediaWiki API: ask for the article as plain text ("extracts" + "explaintext").
# This beats scraping HTML: we get clean prose paragraphs with no nav junk.
# "redirects=1" resolves alternate spellings to the canonical article.
API = (
    "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext"
    f"&titles={TITLE}&format=json&redirects=1"
)

# Wikipedia rejects requests with the default python-urllib User-Agent
# (HTTP 403), so identify ourselves like the API etiquette guidelines ask.
HEADERS = {"User-Agent": "ragdepot-demo/0.1 (local RAG demo)"}

# The depot root: one directory holding one sqlite index per scope.
# Here that's demo/depot-data/global/depot.db ("global" = shared corpus scope).
ROOT = Path(__file__).parent / "depot-data"

# Example questions. Each is run through hybrid search:
#   - dense: meaning similarity via the embedding model (catches paraphrases)
#   - BM25:  keyword scoring over the chunk text (catches exact names/codes)
#   both rankings are fused with reciprocal rank fusion into one list.
QUERIES = [
    "first emperor of Rome",
    "fall of the Western Roman Empire",
    "Pax Romana",
    "Colosseum gladiator games",
]


# -- step 1: get the document ----------------------------------------------


def fetch_article() -> str:
    """Fetch the article as plain text, with MediaWiki headings → markdown.

    The MediaWiki plain-text extract marks sections like:

        == History ==          (level 2)
        === Crisis ===         (level 3)

    ragdepot's chunker only recognises markdown-style ("ATX") headings
    (`## History`), so we translate: a heading marker of N `=`s becomes N+1
    `#`s (`== History ==` -> `### History`; the exact level doesn't matter —
    the chunker reads only the heading text). That matters because each chunk
    keeps its nearest preceding heading — so search hits come back labelled
    with their section.
    """
    req = urllib.request.Request(API, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        # The response JSON nests pages under their page ids; a single-title
        # query has exactly one — take whatever comes first.
        page = next(iter(json.load(r)["query"]["pages"].values()))

    # `re.MULTILINE` makes ^/$ anchor per line, so we rewrite every heading
    # line: "=== X ===" -> "#### X". The backreference \1 requires the same
    # number of '=' on both sides, so stray '=' in prose is left alone.
    return re.sub(
        r"^(=+) ?(.*?) ?\1+$",
        lambda m: f"{'#' * (len(m[1]) + 1)} {m[2]}",
        page["extract"],
        flags=re.MULTILINE,
    )


# -- steps 2-4: store, search, report ----------------------------------------


def main() -> int:
    try:
        # Fetch the source document. This is demo plumbing — the library
        # itself does no network I/O and only ever sees the text.
        text = fetch_article()

        # Depot(root=...) is cheap and pure: it never touches disk, so a
        # broken root doesn't fail here — errors surface on first use,
        # wrapped in DepotError (the "graceful degradation" contract).
        depot = Depot(root=ROOT)

        # Index the article into the "global" scope (the shared corpus —
        # as opposed to the local scopes upsert_file uses for files).
        # Inside one sqlite transaction, ragdepot will:
        #   1. chunk the text  (~800 words each, headings attached,
        #      short nav fragments dropped)
        #   2. embed each chunk with the local default model
        #   3. store chunks + vectors + a full-text (FTS5) index
        # Returns True if the content was new/changed, False if the
        # content hash matched an existing doc (no-op).
        url = f"https://en.wikipedia.org/wiki/{TITLE}"
        added = depot.upsert_text(scope="global", title="Roman Empire", url=url, text=text)

        # stats(): per-scope {docs, chunks} counts — handy sanity check.
        print(
            f"[index] {'indexed' if added else 'unchanged'} "
            f"({depot.stats()['global']['chunks']} chunks)\n"
        )

        # Query time. Each call embeds the question once, then per scope:
        #   - cosine top-N over stored vectors  (dense branch)
        #   - BM25 top-N over the FTS5 index     (keyword branch)
        #   - both fused via RRF, local scopes ranked above "global" on ties
        # Each Hit carries: scope, doc_id, title, path/url, snippet
        # (the chunk text, prefixed by its heading) and the fused score.
        # The demo prints every field of every hit — this is the complete
        # result a RAG client receives, including the provenance (title,
        # path, url) it needs to attribute answers to sources.
        for q in QUERIES:
            hits = depot.query(q, scope="global", k=2)
            print(f"[query] {q}")
            for i, h in enumerate(hits, 1):
                snippet = " ".join(h.snippet.split())[:220]  # collapse whitespace, truncate
                print(f"  --- hit {i} ---")
                print(f"  scope:   {h.scope}")
                print(f"  doc_id:  {h.doc_id}")
                print(f"  title:   {h.title}")
                print(f"  path:    {h.path or '(none)'}")
                print(f"  url:     {h.url or '(none)'}")
                print(f"  score:   {h.score:.4f}")
                print(f"  snippet: {snippet}...")
            print()

    # Everything client-facing fails as DepotError: one type to catch,
    # whether the store is corrupt, the scope is missing, or the embedder
    # blew up. A real client would degrade ("knowledge base unavailable").
    except DepotError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    # Network problems from our own fetch (DNS, timeout, HTTP errors).
    except OSError as e:
        print(f"cannot fetch article: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
