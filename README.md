# ragdepot

A small, standalone hybrid RAG library: chunk text, embed it locally, and
search it with fused dense + BM25 retrieval over per-scope sqlite indexes.
Usable from Python or the CLI — no server, no daemon, no LLM.

## What it does

- **Scopes** — one `global` scope (shared corpus) plus any number of local
  scopes (project, wiki, notebook, ...); each is its own index under the
  depot root. `query(scope=None)` searches everything, with local scopes
  ranked above `global` on score ties
- **Chunking** — rule-based: blank-line paragraphs packed into ~800-word
  chunks, each tagged with its nearest heading; short fragments (nav junk)
  dropped
- **Embedding** — local [fastembed](https://github.com/qdrant/fastembed)
  models; default `minishlab/potion-base-8M` (~30 MB, downloaded once to the
  user cache, fully offline after). Any supported model via `Depot(model=...)`
  / CLI `--model`
- **Hybrid search** — dense cosine top-N + FTS5 BM25 top-N, fused with
  reciprocal rank fusion: paraphrases and exact names/codes both hit
- **Storage** — one sqlite db per scope (`<root>/<scope>/depot.db`), WAL
  mode, vectors as float32 BLOBs, content-hashed idempotent upserts in a
  single transaction
- **Graceful degradation** — `Depot(root)` never touches disk; every
  client-facing failure raises one type, `DepotError`, so callers can answer
  "knowledge base unavailable" instead of crashing

## Requirements

- **Python** ≥ 3.14
- **Disk** — that's it; the only network call is the one-time model download

## Setup

```bash
# 1. Clone and install:
git clone https://github.com/Origamify/ragdepot.git
cd ragdepot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # runtime dep: fastembed (pinned)
pip install -e .
pip install pytest ruff           # dev tools (./test.sh, lint)

# 2. Test:
./test.sh                         # first run downloads the embedding model once
```

`run.sh` / `test.sh` activate `./.venv` automatically when present, then run
`python -m ragdepot` / `pytest`.

## Using it

Python:

```python
from ragdepot import Depot

depot = Depot(root="depot-data")            # cheap and pure: no disk I/O here
depot.upsert_file(scope="notes", path="solar.md")          # idempotent
depot.upsert_text(scope="global", title="Roman Empire",
                  url="https://en.wikipedia.org/wiki/Roman_Empire", text=...)

hits = depot.query("how to store solar energy", scope="notes", k=8)
# Hit fields: scope, doc_id, title, path, url, snippet, score
```

CLI (operators; `query`/`stats` also take `--json`):

```bash
./run.sh --root depot-data add --scope notes solar.md ...
./run.sh --root depot-data query --scope notes "solar energy storage"
./run.sh --root depot-data stats
```

Flags, output formats, and the full API: [interface.md](interface.md).

## Demos

Run from the repo root with the venv active:

- **End-to-end** — `python demo/demo.py` — fetches the Wikipedia article
  "Roman Empire", chunks/embeds/indexes it, runs fact queries, prints every
  `Hit` field. Data lands in gitignored `demo/depot-data/`
- **Model comparison** — `python demo/embed_model_compare.py` — benchmarks
  fastembed models (dense branch + the real hybrid pipeline) on a fixed
  ground-truth corpus with semantic/lexical query tracks; writes a markdown
  report for review (`demo/model-compare-report.md`, gitignored)

## Docs

- `about.md` — short "what is this" page
- `interface.md` — Python API + CLI reference, storage format
- `project.md` — maintainer's map: setup, file map, conventions
- `rag-spec.md` — the spec the library implements

## Notes

- **One model per depot root** — stored vectors are only comparable if docs
  and queries share a model; switching models means re-indexing
- **Thread-safe** — WAL mode and one connection per operation; queries and
  indexing may run at the same time
- The library does no network I/O and reads no files the caller didn't pass
  (the demos fetch their own data)

## License

[MIT](LICENSE)
