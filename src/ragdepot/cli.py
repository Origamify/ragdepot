"""CLI (rag-spec.md §CLI): add / query / stats for operators; query and stats
also emit --json for scripting and non-Python clients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragdepot.depot import Depot, DepotError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragdepot", description="Hybrid RAG depot: dense + BM25 search."
    )
    parser.add_argument("--root", default=".", help="depot root directory")
    parser.add_argument(
        "--model", default=None, help="fastembed model name (default: potion-base-8M)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="index files into a scope")
    p_add.add_argument("--scope", required=True)
    p_add.add_argument("files", nargs="+")

    p_query = sub.add_parser("query", help="search the depot")
    p_query.add_argument("--scope", default=None, help="scope name, or omit for all")
    p_query.add_argument("--json", action="store_true", dest="as_json")
    p_query.add_argument("text")

    p_stats = sub.add_parser("stats", help="per-scope doc/chunk counts")
    p_stats.add_argument("--json", action="store_true", dest="as_json")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        depot = Depot(root=args.root, model=args.model)
        if args.command == "add":
            for f in args.files:
                added = depot.upsert_file(scope=args.scope, path=Path(f))
                print(f"{'indexed' if added else 'unchanged'}: {f}")
        elif args.command == "query":
            hits = depot.query(args.text, scope=args.scope)
            if args.as_json:
                print(json.dumps([h.__dict__ for h in hits], indent=2))
            else:
                for h in hits:
                    print(f"{h.title} [{h.scope}] score={h.score:.4f}")
                    loc = h.path or h.url
                    print(f"  {loc}\n  {h.snippet}\n")
        elif args.command == "stats":
            s = depot.stats()
            print(json.dumps(s, indent=2) if args.as_json else s)
    except DepotError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
