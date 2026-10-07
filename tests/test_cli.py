"""CLI wiring tests: argv -> Depot calls -> output shaping."""

import json

from ragdepot import cli
from ragdepot.depot import Hit


class FakeDepot:
    instances: list[FakeDepot] = []

    def __init__(self, root, model=None):
        self.root = root
        self.model = model
        self.calls = []
        FakeDepot.instances.append(self)

    def upsert_file(self, scope, path):
        self.calls.append(("upsert_file", scope, str(path)))
        return True

    def query(self, text, scope=None, k=8):
        self.calls.append(("query", text, scope))
        return [
            Hit("s", "d1", "Title One", "a.md", "", "the snippet text", 0.042),
            Hit("global", "d2", "Title Two", "", "https://x", "other snippet", 0.031),
        ]

    def stats(self):
        return {"s": {"docs": 1, "chunks": 2}}


def run(monkeypatch, *argv):
    FakeDepot.instances.clear()
    monkeypatch.setattr(cli, "Depot", FakeDepot)
    rc = cli.main(list(argv))
    depot = FakeDepot.instances[0]
    return rc, depot


def test_add_wiring(monkeypatch, capsys):
    rc, depot = run(monkeypatch, "--root", "/tmp/r", "add", "--scope", "wiki", "a.md", "b.md")
    assert rc == 0 and depot.calls == [
        ("upsert_file", "wiki", "a.md"),
        ("upsert_file", "wiki", "b.md"),
    ]
    assert "indexed: a.md" in capsys.readouterr().out


def test_model_flag_wiring(monkeypatch):
    rc, depot = run(monkeypatch, "--root", "/tmp/r", "--model", "m/x", "stats")
    assert rc == 0 and depot.model == "m/x"


def test_query_json(monkeypatch, capsys):
    rc, depot = run(
        monkeypatch, "--root", "/tmp/r", "query", "--json", "--scope", "wiki", "panel output"
    )
    out = capsys.readouterr().out
    assert rc == 0
    parsed = json.loads(out)
    assert parsed[0]["title"] == "Title One" and parsed[1]["url"] == "https://x"
    assert depot.calls == [("query", "panel output", "wiki")]


def test_query_text_blocks(monkeypatch, capsys):
    rc, _ = run(monkeypatch, "--root", "/tmp/r", "query", "panel output")
    out = capsys.readouterr().out
    assert rc == 0 and "Title One [s]" in out and "a.md" in out and "the snippet text" in out


def test_stats_json(monkeypatch, capsys):
    rc, _ = run(monkeypatch, "--root", "/tmp/r", "stats", "--json")
    assert rc == 0 and json.loads(capsys.readouterr().out) == {"s": {"docs": 1, "chunks": 2}}


def test_depot_error_exit_code(monkeypatch, capsys):
    class BadDepot(FakeDepot):
        def stats(self):
            from ragdepot.depot import DepotError

            raise DepotError("knowledge base unavailable")

    monkeypatch.setattr(cli, "Depot", BadDepot)
    rc = cli.main(["--root", "/tmp/r", "stats"])
    assert rc == 1 and "unavailable" in capsys.readouterr().err
