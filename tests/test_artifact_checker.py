# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""``tools/check_benchmark_artifact.py`` accepts a consistent artifact and rejects broken ones."""

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

import benchmark as B

ROOT = Path(__file__).resolve().parent.parent
SHA = "9fd5ccfca1789da6b58c7001264cd362efe4e92c"


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_benchmark_artifact", ROOT / "tools" / "check_benchmark_artifact.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_table(configs, splits, seeds, mem, validation):
    n = len(list(splits))
    return {
        name: {
            "acc": [0.8 + 0.001 * i for i in range(n)],
            "forgetting": [0.05 if name == "ordinary" else 0.02] * n,
            "plasticity": [0.8] * n,
            "newest": [0.5] * n,
            "old": [0.8] * n,
            "immediate": [[0.9] * 5] * n,
        }
        for name in configs
    }


@pytest.fixture
def artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "table", fake_table)
    monkeypatch.setattr(
        B,
        "provenance",
        lambda: {
            "commit": SHA,
            "dirty": False,
            "python": "3",
            "numpy": "1",
            "scipy": "1",
            "scikit-learn": "1",
            "protocol": {},
        },
    )
    with contextlib.redirect_stdout(io.StringIO()):
        B.main(str(tmp_path / "benchmark_raw.json"))
    write_report(tmp_path)
    return tmp_path


def write_report(directory):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        B.report(json.loads((directory / "benchmark_raw.json").read_text()))
    (directory / "benchmark_report.txt").write_text(buf.getvalue())


def run(checker, directory, *extra):
    old = sys.argv
    sys.argv = ["check", str(directory), *extra]
    try:
        checker.main()
    finally:
        sys.argv = old


def edit(directory, fn):
    path = directory / "benchmark_raw.json"
    raw = json.loads(path.read_text())
    fn(raw)
    path.write_text(json.dumps(raw))


def test_consistent_artifact_passes(artifact):
    run(load_checker(), artifact, "--expect-commit", SHA)


def test_wrong_commit_is_rejected(artifact):
    with pytest.raises(SystemExit):
        run(load_checker(), artifact, "--expect-commit", "1" * 40)


def test_dirty_provenance_is_rejected(artifact):
    edit(artifact, lambda r: r["provenance"].update(dirty=True))
    write_report(artifact)
    with pytest.raises(SystemExit):
        run(load_checker(), artifact)


def test_missing_split_is_rejected(artifact):
    edit(artifact, lambda r: r["test_mem60"]["relaxed"]["acc"].pop())
    with pytest.raises(SystemExit):
        run(load_checker(), artifact)


def test_report_that_does_not_match_the_raw_file_is_rejected(artifact):
    edit(artifact, lambda r: r["test_mem60"]["relaxed"]["acc"].__setitem__(3, 0.99))
    with pytest.raises(SystemExit):  # raw changed, benchmark_report.txt left as it was
        run(load_checker(), artifact)
