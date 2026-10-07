# ragdepot — interface.md

## Python API

```python
from ragdepot import Depot, Hit, DepotError

depot = Depot(root=..., model=...)           # no disk access at construction
depot.upsert_file(scope="notes", path=...)
depot.upsert_text(scope="global", title=..., url=..., text="...")
hits = depot.query("text", scope=None, k=8)  # scope: name | "global" | None=all
depot.delete_doc(doc_id)
depot.delete_file(scope, path)
depot.stats()                            # {scope: {"docs": n, "chunks": m}}
```

`Hit` fields: `scope, doc_id, title, path, url, snippet, score` (frozen
dataclass; snippet = `[heading] chunk text`). All failures raise `DepotError`.
`Depot(root, embed_fn=...)` lets tests inject a deterministic embedder.
`model` selects any fastembed-supported embedding model (default:
`minishlab/potion-base-8M`). One model per depot root: stored vectors are only
comparable if docs and queries share a model, so switching means re-indexing.

## CLI

```
python -m ragdepot [--root DIR] [--model NAME] add   --scope SCOPE FILE...  # index
python -m ragdepot [--root DIR] [--model NAME] query --scope SCOPE [--json] TEXT
python -m ragdepot [--root DIR] [--model NAME] stats [--json]
```

- `--root DIR`: depot root directory (default: current directory).
- `--model NAME`: fastembed model name (default: `minishlab/potion-base-8M`).
- Exit codes: 0 ok, 1 depot error (message on stderr), 2 usage (argparse).
- `add` prints `indexed: FILE` or `unchanged: FILE` per file.
- `query` text output: one block per hit — `title [scope] score`, location
  (path or url), snippet. `--json`: list of Hit objects.
- `stats --json`: `{"scope": {"docs": n, "chunks": m}}`; scopes with no
  docs are omitted.

## Storage format (per scope)

`<root>/<scope>/depot.db` — sqlite, WAL mode: `chunks` table (id, doc_id,
title, path, url, heading, text, vector float32 BLOB) + FTS5 virtual table
over chunk text for BM25. Managed by the library; not a public interface.
