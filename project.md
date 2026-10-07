# ragdepot — project.md

## Status

Active. Library complete (chunker, embedder, per-scope storage, hybrid
search, facade, CLI). Spec: `rag-spec.md`.

## Scope

- Does: rule-based chunking, one default local embedder (fastembed/potion-base-8M,
  selectable via `model=` / `--model`), per-scope
  sqlite/FTS5 storage, hybrid dense+BM25 search with rank fusion, CLI,
  `DepotError` degradation contract.
- Doesn't: reranking, query expansion, LLM use, server/daemon, dump-format
  parsers, client-side integration.

## Run & setup

- Create the project venv and install (Python ≥ 3.14):

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt   # runtime dep: fastembed (pinned)
  pip install -e .                  # ragdepot itself, editable
  pip install pytest ruff           # dev tools (./test.sh, lint)
  ```

- `run.sh` and `test.sh` activate `./.venv` automatically when present;
  otherwise they fall back to the ambient environment.
- Runtime dep: `fastembed` (pinned in `requirements.txt`). First embed
  downloads the model (~30 MB) to the user cache; after that it is offline.
- Tests: `./test.sh` (runs everything; the real-model embedder test downloads
  the model once on first run).
- Lint: `ruff check src tests` + `ruff format --check src tests`.

## File map

- `src/ragdepot/depot.py` — public facade (`Depot`); to change the API → here.
- `src/ragdepot/types.py` — `DepotError`, `Hit` (leaf module, no cycles).
- `src/ragdepot/chunker.py` — paragraph chunking (~800 words, headings, junk drop).
- `src/ragdepot/embedder.py` — fastembed model (selectable, lazy load), unit vectors.
- `src/ragdepot/index.py` — per-scope sqlite store: FTS5 BM25, vector BLOBs,
  WAL, atomic upserts.
- `src/ragdepot/search.py` — dense + BM25 candidates, RRF fusion, local-first merge.
- `src/ragdepot/cli.py`, `__main__.py` — add/query/stats CLI with `--json`.
- `tests/` — pytest; `conftest.py` has the deterministic fake embedder used
  by the unit tests.
- `run.sh`, `test.sh` — wrappers that activate `./.venv` when present, then
  run `python -m ragdepot` / `pytest`.
- `demo/demo.py` — end-to-end demo: fetches the Roman Empire Wikipedia article,
  indexes it, runs fact queries (data in `demo/depot-data/`, gitignored).
- `demo/embed_model_compare.py` — benchmarks fastembed models on a fixed
  ground-truth corpus; writes a markdown report for AI review
  (`demo/model-compare-report.md`, gitignored).

## Conventions

- All client-facing failures raise `DepotError` (graceful degradation contract).
- `Depot(root=...)` never touches disk; errors surface on first use.
- Writes are atomic per scope (one sqlite transaction per upsert). Local docs
  are keyed by resolved absolute path (identity, not raw string). Files with no
  indexable chunks (e.g. headings-only) store nothing and clear any stale doc.
- No network I/O; reads only files the caller passes.
- Tests inject `embed_fn` (fake in `conftest.py`); one embedder test uses the
  real model (downloaded once on first run).
- Scope names must match `[A-Za-z0-9][A-Za-z0-9_-]*` (they become dir names).
- Storage layout: `<root>/<scope>/depot.db` (WAL mode).

## Settings

None. No env vars read by the code. Depot root always comes from the caller
(CLI `--root`, default `.`).
