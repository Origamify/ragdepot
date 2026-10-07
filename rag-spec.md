# ragdepot — RAG module spec

ragdepot is a small, standalone RAG library. It works with any caller:
text and paths go in, search hits come out. It makes no assumptions about
the host application.

## Goals

- One **global scope** (e.g. shared reference articles) available to all queries.
- **Local scopes**: one index per corpus (project, wiki, notebook, ...).
- Hybrid (dense + keyword) search over both.

Out of scope: dump-format parsers, any server or daemon.
Library + CLI only. Client-side integration is separate work.

## Concepts

| Term | Meaning |
|---|---|
| depot root | One directory holding all indexes |
| scope | `global` (shared corpus) or a named local index |
| document | One indexed unit: a file path (local) or a title + URL (global) |
| chunk | A slice of a document; embedded and searched as one unit |

## Retrieval

Every query runs two searches and merges the results:

- **Dense**: embedding similarity. One default embedding model
  (`minishlab/potion-base-8M`); callers may select any fastembed-supported
  model via `Depot(model=...)` / CLI `--model`. One model per depot root —
  switching models requires re-indexing.
- **BM25**: keyword scoring over the same chunks. Catches exact names,
  codes, and IDs that vectors miss.
- The two ranked lists are fused (reciprocal rank fusion or similar) into
  one final ranking.

### Chunking

Rule-based, no LLM. Split on paragraph boundaries into chunks of roughly
800 words. Each chunk keeps its nearest heading (so context survives).
Very short fragments (navigation junk) are dropped.

## Python interface

```python
from ragdepot import Depot

depot = Depot(root=...)

# add content — idempotent: unchanged content is a no-op,
# changed content replaces its old chunks
depot.upsert_file(scope="notes", path=...)
depot.upsert_text(scope="global", title=..., url=..., text="...")

hits = depot.query("how to store solar energy", scope="notes", k=8)
# scope: "global" | a local scope name | None (search everything, local first)

depot.delete_doc(doc_id)
depot.delete_file(scope, path)   # when a source is removed
depot.stats()                    # per-scope doc/chunk counts
```

`Hit` fields: `scope, doc_id, title, path, url, snippet, score`. The
snippet is the chunk text.

## CLI

```
python -m ragdepot add   --scope global file.md ...
python -m ragdepot query --scope notes "solar energy storage"
python -m ragdepot stats
```

For operators; nothing else depends on it. One machine-facing extra:
`query` and `stats` accept `--json` so non-Python clients can consume
hits without parsing text output.

## Requirements

- No network I/O; reads no files the caller didn't pass.
- Thread-safe: queries and indexing may run at the same time.
- The embedding model is downloaded once on first use (tests included).
