#!/usr/bin/env python
"""Compare fastembed models on a fixed retrieval benchmark; write a markdown
report designed for AI review (or human skimming).

Per model, over two query tracks (see embed_bench_data.py):
    - semantic track: paraphrase questions, no lexical overlap — the
      dense/embedding model's home ground;
    - lexical track: short keyword queries where every token appears in the
      target doc — BM25's home ground.

For each model it records, per track: Hit@1 / Hit@3 / MRR of the dense
branch (embed query -> cosine top-N, the library's own code path), plus a
**hybrid** score: the same queries through the library's real `search()`
(dense + BM25 fused with reciprocal rank fusion) — the pipeline ragdepot
actually ships. Index time and warm query time round out the picture.

The BM25 baseline is scored with the library's real FTS5 branch.

The output is deterministic markdown: summary table sorted by overall dense
MRR, per-track rank tables, mechanical findings, and a "how to review"
header so an AI reader knows exactly what each number means.

Usage (from the repo root, venv with ragdepot + fastembed installed):
    python demo/embed_model_compare.py                 # default model set
    python demo/embed_model_compare.py --models A B    # explicit models
    python demo/embed_model_compare.py --out report.md

First run downloads each model once (~30-440 MB each) into the user cache;
after that everything is offline. A failing model (bad name, no network,
corrupt download) becomes a FAILED row; other models still run.
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from embed_bench_data import CORPUS, LEXICAL_QUERIES, SEMANTIC_QUERIES

from ragdepot.embedder import MODEL_NAME, embed
from ragdepot.index import Store, content_hash, doc_id_for
from ragdepot.search import dense_top, search

# Small, permissively licensed, fastembed-supported. Filtered against
# TextEmbedding.list_supported_models() at runtime; --models overrides.
DEFAULT_MODELS = [
    "minishlab/potion-base-8M",  # ragdepot default (static embeddings; baseline)
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "snowflake/snowflake-arctic-embed-xs",
    "snowflake/snowflake-arctic-embed-s",
    "snowflake/snowflake-arctic-embed-m",
    "sentence-transformers/all-MiniLM-L6-v2",
    "nomic-ai/nomic-embed-text-v1.5-Q",
    "thenlper/gte-base",
    "jinaai/jina-embeddings-v2-small-en",
    "minishlab/potion-retrieval-32M",
]

SCOPE = "bench"

REPORT_HEADER_FOR_REVIEWER = """\
## How to review this report

- Two query tracks: **semantic** (paraphrases, no shared vocabulary with the
  answer — what embeddings are for) and **lexical** (short keyword queries
  where every token appears in the answer doc — what keyword search is for).
- Columns are per-track MRR of the dense branch; **hybrid MRR** runs the same
  queries through the library's real pipeline (dense + BM25, reciprocal rank
  fusion). A hybrid score below its own dense score means fusion is hurting
  that model; below BM25's lexical score on lexical queries means the dense
  branch is dragging it down.
- **Hit@1 / Hit@3**: fraction of queries where the expected document ranks
  first / in the top 3. **MRR**: mean of 1/rank (miss = 0). All 0-1,
  higher is better. `–` in a rank cell means the expected doc was not found.
- **index ms**: model load (warm, after any download) + embed all corpus
  docs + write store. **query ms**: mean per query of embed + search, model
  warm. Single-run timings, one host — treat as rough, not benchmark-grade.
- The BM25 row is the library's real FTS5 branch, which ANDs every token:
  it should score ~1.0 on the lexical track and ~0.0 on the semantic track.
  That split is the entire reason the library fuses the two branches.
- Semantic queries marked *(trap)* carry distractor vocabulary from a
  different document; per-query ranks show whether a model falls for them.
- The corpus is small ({n_docs} docs): MRR differences under ~0.05 are
  noise, and a perfect score means the benchmark cannot separate the top
  models — judge by per-query ranks and trap handling, not the winner alone.
"""


# -- machinery ----------------------------------------------------------------


@dataclass
class Result:
    model: str
    dim: int = 0
    size_gb: float | None = None
    index_ms: float = 0.0
    query_ms: float = 0.0
    sem_ranks: list[int | None] = field(default_factory=list)
    lex_ranks: list[int | None] = field(default_factory=list)
    hybrid_ranks: list[int | None] = field(default_factory=list)  # all queries
    error: str | None = None

    @staticmethod
    def _mrr(ranks: list[int | None]) -> float:
        return sum(1 / r if r else 0.0 for r in ranks) / len(ranks) if ranks else 0.0

    @staticmethod
    def _hit1(ranks: list[int | None]) -> float:
        return sum(1 for r in ranks if r == 1) / len(ranks) if ranks else 0.0

    @property
    def mrr_sem(self) -> float:
        return self._mrr(self.sem_ranks)

    @property
    def mrr_lex(self) -> float:
        return self._mrr(self.lex_ranks)

    @property
    def mrr_all(self) -> float:
        return self._mrr(self.sem_ranks + self.lex_ranks)

    @property
    def hybrid_mrr(self) -> float:
        return self._mrr(self.hybrid_ranks)


def _model_sizes() -> dict[str, float]:
    from fastembed import TextEmbedding

    try:
        return {m["model"]: m["size_in_GB"] for m in TextEmbedding.list_supported_models()}
    except Exception:
        return {}


def _supported(models: list[str]) -> list[str]:
    """Default set filtered to models this fastembed actually supports."""
    from fastembed import TextEmbedding

    names = {m["model"] for m in TextEmbedding.list_supported_models()}
    return [m for m in models if m in names]


def _build_store(model: str, root: Path) -> tuple[Store, float]:
    """One store for the corpus, embedded with `model`. Returns (store, ms)."""
    embed(["warm up"], model=model)  # load (and download) before timing starts
    t0 = time.perf_counter()
    vectors = embed([text for _, _, text in CORPUS], model=model)
    store = Store(root, SCOPE)
    for (doc_id, title, text), vec in zip(CORPUS, vectors, strict=True):
        store.upsert(
            doc_id_for(SCOPE, doc_id),
            title=title,
            path="",
            url="",
            text_hash=content_hash(text),
            chunks=[(title, text, vec.tobytes())],
        )
    return store, (time.perf_counter() - t0) * 1000


def _rank_of_expected(ordered_doc_ids: list[str], expected: str) -> int | None:
    return ordered_doc_ids.index(expected) + 1 if expected in ordered_doc_ids else None


def _doc_ids_in_order(store: Store, chunk_ids: list[int]) -> list[str]:
    rows = store.fetch(chunk_ids)
    id_to_doc = {doc_id_for(SCOPE, d[0]): d[0] for d in CORPUS}
    return [id_to_doc[rows[c][1]] for c in chunk_ids if c in rows]


def bench_model(model: str) -> Result:
    """Dense and hybrid retrieval metrics for one model over both tracks."""
    res = Result(model=model)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            store, res.index_ms = _build_store(model, Path(tmp))
            k = len(CORPUS)
            times, sem, lex, hyb = [], [], [], []
            for track, out in ((SEMANTIC_QUERIES, sem), (LEXICAL_QUERIES, lex)):
                for query, expected, _ in track:
                    t0 = time.perf_counter()
                    qvec = embed([query], model=model)[0]
                    # dense branch: cosine top-N (same code path as the library)
                    dense = _doc_ids_in_order(store, [c for c, _ in dense_top(store, qvec, k)])
                    out.append(_rank_of_expected(dense, expected))
                    # hybrid: the shipped pipeline (dense + BM25, RRF fusion)
                    hits = search([(SCOPE, store)], query, qvec, k)
                    hyb.append(
                        _rank_of_expected([h.doc_id for h in hits], doc_id_for(SCOPE, expected))
                    )
                    times.append((time.perf_counter() - t0) * 1000)
            res.dim = len(store.vectors()[1][0])
            res.query_ms = sum(times) / len(times)
            res.sem_ranks, res.lex_ranks, res.hybrid_ranks = sem, lex, hyb
    except Exception as e:  # bad name, download failure, OOM — record and move on
        res.error = f"{type(e).__name__}: {e}"[:120]
    return res


def bench_bm25() -> Result:
    """Keyword baseline: the library's real FTS5 branch on both tracks."""
    out = Result(model="BM25 (FTS5 baseline, no embeddings)")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = _build_store(MODEL_NAME, Path(tmp))
            for track in (SEMANTIC_QUERIES, LEXICAL_QUERIES):
                ranks = []
                for query, expected, _ in track:
                    hits = store.bm25(query, len(CORPUS))
                    ordered = _doc_ids_in_order(store, [c for c, _ in hits])
                    ranks.append(_rank_of_expected(ordered, expected))
                out.sem_ranks += [] if track is LEXICAL_QUERIES else ranks
                out.lex_ranks += [] if track is SEMANTIC_QUERIES else ranks
    except Exception as e:
        out.error = f"{type(e).__name__}: {e}"[:120]
    return out


# -- report -------------------------------------------------------------------


def _fmt_rank(r: int | None) -> str:
    return "–" if r is None else str(r)


def _rank_table(
    title: str,
    queries: list,
    results: list[Result],
    baseline_ranks: list[int | None],
    attr: str,
) -> list[str]:
    lines = [f"## {title}", ""]
    header = "| # | query | expected |" + "".join(f" {r.model.split('/')[-1]} |" for r in results)
    header += " bm25 |"
    lines.append(header)
    lines.append("|" + "---|" * (3 + len(results) + 1))
    for i, (query, expected, trap) in enumerate(queries):
        cells = "".join(f" {_fmt_rank(getattr(r, attr)[i])} |" for r in results)
        cells += f" {_fmt_rank(baseline_ranks[i])} |"
        note = " *(trap)*" if trap else ""
        lines.append(f"| {i + 1} | {query}{note} | {expected} |{cells}")
    lines.append("")
    return lines


def render(results: list[Result], baseline: Result, meta: list[str]) -> str:
    lines = ["# ragdepot embedding-model comparison", ""]
    lines += [f"- {m}" for m in meta]
    lines += [
        f"- corpus: {len(CORPUS)} docs (1 chunk each); queries: "
        f"{len(SEMANTIC_QUERIES)} semantic + {len(LEXICAL_QUERIES)} lexical, "
        "both with ground truth",
        "",
    ]
    lines += [REPORT_HEADER_FOR_REVIEWER.format(n_docs=len(CORPUS)).rstrip(), ""]

    ok = sorted([r for r in results if not r.error], key=lambda r: -r.mrr_all)
    failed = [r for r in results if r.error]

    lines.append("## Summary (sorted by overall dense MRR, higher is better)")
    lines.append("")
    lines.append(
        "| model | dim | size GB | query ms | MRR lex | MRR sem | MRR all "
        "| Hit@1 all | hybrid MRR |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in ok:
        size = f"{r.size_gb:.2f}" if r.size_gb is not None else "?"
        lines.append(
            f"| {r.model} | {r.dim} | {size} | {r.query_ms:.1f} "
            f"| {r.mrr_lex:.3f} | {r.mrr_sem:.3f} | {r.mrr_all:.3f} "
            f"| {Result._hit1(r.sem_ranks + r.lex_ranks):.2f} | {r.hybrid_mrr:.3f} |"
        )
    lines.append(
        f"| {baseline.model} | – | – | – | {baseline.mrr_lex:.3f} "
        f"| {baseline.mrr_sem:.3f} | {baseline.mrr_all:.3f} "
        f"| {Result._hit1(baseline.sem_ranks + baseline.lex_ranks):.2f} | – |"
    )
    for r in failed:
        lines.append(f"| {r.model} | FAILED | {r.error} | | | | | | |")
    lines.append("")

    lines += _rank_table(
        "Lexical track — rank of the expected document (keyword home ground)",
        LEXICAL_QUERIES,
        ok,
        baseline.lex_ranks,
        "lex_ranks",
    )
    lines += _rank_table(
        "Semantic track — rank of the expected document (paraphrase home ground)",
        SEMANTIC_QUERIES,
        ok,
        baseline.sem_ranks,
        "sem_ranks",
    )

    lines.append("## Mechanical findings")
    lines.append("")
    if ok:
        best = ok[0]
        hyb_best = max(ok, key=lambda r: r.hybrid_mrr)
        fusion_wins = sum(1 for r in ok if r.hybrid_mrr >= r.mrr_all)
        fast = min(ok, key=lambda r: r.query_ms)
        lines.append(f"- best overall dense MRR: {best.model} ({best.mrr_all:.3f})")
        lines.append(f"- best hybrid MRR: {hyb_best.model} ({hyb_best.hybrid_mrr:.3f})")
        lines.append(
            f"- fusion helped or tied in {fusion_wins}/{len(ok)} models "
            "(hybrid MRR >= own dense MRR)"
        )
        lines.append(
            f"- BM25: lexical MRR {baseline.mrr_lex:.3f}, semantic {baseline.mrr_sem:.3f} "
            "— the gap hybrid search exists to close"
        )
        lines.append(f"- fastest warm query: {fast.model} ({fast.query_ms:.1f} ms/query)")
    if failed:
        lines.append(
            f"- {len(failed)} model(s) failed to run: " + ", ".join(r.model for r in failed)
        )
    lines.append("")
    return "\n".join(lines)


# -- entry --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark fastembed models for ragdepot; writes a markdown report."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="fastembed model names to compare (default: curated small set)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "model-compare-report.md"),
        help="report path (default: demo/model-compare-report.md)",
    )
    args = parser.parse_args(argv)

    models = args.models if args.models else _supported(DEFAULT_MODELS)
    if not models:
        print("error: no usable models (check --models / fastembed install)", file=sys.stderr)
        return 1

    sizes = _model_sizes()
    n_queries = len(SEMANTIC_QUERIES) + len(LEXICAL_QUERIES)
    print(
        f"benchmarking {len(models)} models x {n_queries} queries (first run downloads models)..."
    )

    results = []
    for m in models:
        print(f"  {m} ...", flush=True)
        r = bench_model(m)
        r.size_gb = sizes.get(m)
        results.append(r)

    baseline = bench_bm25()

    import fastembed

    meta = [
        f"generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"host: python {platform.python_version()}, {platform.system().lower()}, "
        f"fastembed {fastembed.__version__}",
        "benchmark: dense cosine top-N per model + hybrid (dense + BM25, RRF) "
        "via the library's real search(), one run per model",
    ]
    report = render(results, baseline, meta)

    out = Path(args.out)
    out.write_text(report)
    done = sum(1 for r in results if not r.error)
    print(f"wrote {out} ({done}/{len(results)} models ok, bm25 lexical MRR {baseline.mrr_lex:.3f})")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
