# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Check a downloaded ``benchmark-<revision>`` workflow artifact.

Usage::

    python tools/check_benchmark_artifact.py DIR [--expect-commit SHA]

``DIR`` holds the unzipped artifact (``benchmark_raw.json``, ``benchmark_report.txt``).  The check
is independent of the workflow's own steps: it re-reads the raw file, verifies provenance and the
structure of every section, recomputes the paired intervals from the per-split data, and
regenerates the report to compare it with ``benchmark_report.txt``.  It exits non-zero on the
first failure.  Passing means the artifact is internally consistent; it says nothing about the
scientific claims.
"""

import argparse
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import benchmark as B

METRICS = ("acc", "forgetting", "plasticity", "newest", "old")
SPLITS = 10
VALIDATION_CANDIDATES = 25
NO_CONTROL = "m[1,1,1,1]"  # the all-ones baseline in benchmark.controls()


def fail(msg):
    sys.exit(f"FAIL: {msg}")


def check_provenance(raw, expect):
    prov = raw["provenance"]
    if prov.get("dirty") is not False:
        fail(f"provenance.dirty is {prov.get('dirty')!r}, expected False")
    if not re.fullmatch(r"[0-9a-f]{40}", str(prov.get("commit"))):
        fail(f"provenance.commit is not a full SHA: {prov.get('commit')!r}")
    if expect and prov["commit"] != expect:
        fail(f"provenance.commit {prov['commit']} != expected {expect}")
    for key in ("python", "numpy", "scipy", "scikit-learn"):
        if not prov.get(key):
            fail(f"provenance.{key} missing")
    return prov


def check_structure(raw):
    methods = [
        "ordinary",
        *[f"lr x{s}" for s in B.LR_SCALES[1:]],
        "strategy_b",
        "relaxed",
    ]
    for mem in (60, 100):
        sec = raw.get(f"test_mem{mem}")
        if sec is None or list(sec) != methods:
            fail(f"test_mem{mem}: expected methods {methods}, got {list(sec or [])}")
        for name, rows in sec.items():
            for metric in METRICS:
                if len(rows[metric]) != SPLITS:
                    fail(
                        f"test_mem{mem}/{name}/{metric}: {len(rows[metric])} != {SPLITS}"
                    )
    rel = raw["relaxation"]
    for key in ("selected", "config", "ordinary_forgetting", "validation"):
        if key not in rel:
            fail(f"relaxation.{key} missing")
    for base in ("ordinary", "strategy_b", "relaxed"):
        entry = raw["degree_control"][base]
        if len(entry["validation"]) != VALIDATION_CANDIDATES:
            fail(
                f"degree_control/{base}: {len(entry['validation'])} validation candidates"
            )
        # No test comparison exists when nothing was feasible or when the selected candidate is
        # the all-ones baseline (identical to "no control"); otherwise the test arms must exist.
        expects_test = entry["selected"] not in (None, NO_CONTROL)
        if expects_test != (entry["test"] is not None):
            fail(
                f"degree_control/{base}: selection {entry['selected']!r} and test result disagree"
            )
        if entry["test"] is not None:
            for arm in ("base", "ctrl"):
                if len(entry["test"][arm]["acc"]) != SPLITS:
                    fail(f"degree_control/{base}/{arm}: wrong number of splits")


def check_report(raw, report_path):
    buf = io.StringIO()
    with redirect_stdout(buf):
        B.report(raw)
    if buf.getvalue() != report_path.read_text():
        fail(
            "benchmark_report.txt differs from the report regenerated from the raw file"
        )
    # independent recomputation of the paired 95 % t-intervals (not via benchmark.paired_ci)
    for mem in (60, 100):
        sec = raw[f"test_mem{mem}"]
        for a, b in (
            ("strategy_b", "ordinary"),
            ("relaxed", "ordinary"),
            ("relaxed", "strategy_b"),
        ):
            for metric in ("acc", "forgetting"):
                diff = np.asarray(sec[a][metric]) - np.asarray(sec[b][metric])
                half = (
                    stats.t.ppf(0.975, len(diff) - 1)
                    * diff.std(ddof=1)
                    / np.sqrt(len(diff))
                )
                got = B.paired_ci(sec[a][metric], sec[b][metric])
                for x, y in zip(
                    (diff.mean(), diff.mean() - half, diff.mean() + half), got
                ):
                    if abs(x - y) > 1e-9:
                        fail(
                            f"mem{mem} {a} vs {b} {metric}: interval {y} != recomputed {x}"
                        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", type=Path)
    ap.add_argument(
        "--expect-commit", help="revision the artifact must have been built from"
    )
    args = ap.parse_args()
    raw = json.loads((args.directory / "benchmark_raw.json").read_text())
    prov = check_provenance(raw, args.expect_commit)
    check_structure(raw)
    check_report(raw, args.directory / "benchmark_report.txt")
    print(f"OK: artifact for {prov['commit']} is clean, complete and self-consistent")


if __name__ == "__main__":
    main()
