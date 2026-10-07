"""Scaffold checks: package imports, CLI help, facade contract."""

import subprocess
import sys

import ragdepot
from ragdepot import Depot, DepotError, Hit


def test_public_api_importable():
    assert ragdepot.Depot is Depot
    assert ragdepot.Hit is Hit
    assert ragdepot.DepotError is DepotError


def test_depot_construction_touches_no_disk(tmp_path):
    # Broken/absent root must not surface at construction (degradation contract).
    Depot(root=tmp_path / "does" / "not" / "exist")


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "ragdepot", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "add" in result.stdout and "query" in result.stdout and "stats" in result.stdout
