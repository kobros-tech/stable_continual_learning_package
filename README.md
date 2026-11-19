# Stable continual learning package

Class-incremental digits (five experiences) with degree-balanced polynomial features. It
compares ordinary SGD with `strategy_b` (null-space step) under the diagonal degree-control
rate `d_j = eta_deg(j) * m_deg(j)` of the `polynomial_cl` `diagonal_control` PR. The earlier
"null-space" update is `strategy_b` (beta = 0, tol -> 0); `tol > 0` and `beta > 0` are named
relaxations. Mathematics: `math.tex`. Full explanation: `explain.tex`. Both are LaTeX sources;
the PDFs are not committed and are built locally (see below).

## Layout

| path                      | role                                                                                |
| ------------------------- | ----------------------------------------------------------------------------------- |
| `stable_cl/features.py`   | `poly_features`, `expand`: degree-ordered polynomial features, zero-initialised growth |
| `stable_cl/optimization.py` | `diagonal_rates`, `null_basis`, `null_step`, `free_dim`: the rates and the update |
| `stable_cl/training.py`   | `softmax`, `train_task`: replay minibatches and one experience of training          |
| `stable_cl/experiment.py` | `load_digits_features`, `run`, `demo`: data, the five experiences, metrics          |
| `benchmark.py`            | validation tuning, paired test comparisons, degree-control selection                |
| `tests/`                  | pytest suite (numerical properties of every function; `test_assumptions.py` checks A2-A5) |
| `formal/`                 | Lean 4 + Mathlib proofs about the exact update (M0-M5); see `formal/README.md`     |
| `docs/`                   | `traceability.md` (claim, proof, test, experiment), `benchmark_protocol.md`         |
| `math.tex`                | specification: definitions, propositions, proofs, results                           |
| `explain.tex`             | full explanation of the package (features, replay, null-space update, experiments)  |

Every function has a docstring with its purpose, argument shapes, return value and, where it
implements a mathematical operation, the equation (Sphinx `.. math::` syntax). The equations are
the same ones as in `math.tex`.

## Run the code

Requires Python 3.11 or newer with NumPy, SciPy and scikit-learn (`pytest` for the tests).

```bash
python -m pytest                  # test suite (seconds)
python -m stable_cl    # demo: three methods on one split
python benchmark.py --out results/benchmark_raw.json  # several minutes on one CPU core
python benchmark.py --report results/benchmark_raw.json  # recompute comparisons from raw results
```

One test compares a `strategy_b` step with the PR implementation. It is skipped, with the reason
shown by `pytest -rs`, unless the importable `polynomial_cl` is the `diagonal_control` PR version
(it needs `apply_degree_control`, `step_scale_for`, `degree_of_feature`, `fit_task`,
`null_space_basis`), e.g.
`PYTHONPATH=/path/to/polynomial_cl_core-diagonal_control/src python -m pytest tests/test_pr_parity.py`.

## Build the PDFs

Needs a TeX distribution with `pdflatex` and the packages `lmodern`, `microtype`, `amsmath`,
`amsthm`, `mathtools`, `enumitem`, `booktabs`, `hyperref` (all in TeX Live and MiKTeX), plus
those loaded by `explain.tex`. Run each file twice so cross-references resolve:

```bash
pdflatex math.tex && pdflatex math.tex
pdflatex explain.tex && pdflatex explain.tex
```

or `latexmk -pdf math.tex explain.tex`. Delete the `.aux`, `.log` and `.out` files afterwards if
you do not want them in the repository.

## Keeping the documents in sync

The numbers in section 5 of `math.tex` and in `explain.tex` come from `python benchmark.py` and
`python -m stable_cl`; if the code or the protocol changes, rerun them, update those
numbers, and rebuild the PDFs.
