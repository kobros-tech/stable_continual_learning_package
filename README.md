# Stable continual learning package

Class-incremental digits (five experiences) with degree-balanced polynomial features. It
compares ordinary SGD with `strategy_b` (null-space step) under the diagonal degree-control
rate `d_j = eta_deg(j) * m_deg(j)` of the `polynomial_cl` `diagonal_control` PR. The earlier
"null-space" update is `strategy_b` (beta = 0, tol -> 0); `tol > 0` and `beta > 0` are named
relaxations. Mathematics: `math.pdf`. Full explanation: `explain.pdf`.

## Files

| file | role |
|---|---|
| `stable_continual_learning.py` | features, diagonal rates, the two updates, training loop, demo, `--check` |
| `benchmark.py` | validation tuning, paired test comparisons, degree-control selection |
| `math.tex`, `math.pdf` | specification: definitions, propositions, proofs, results |
| `explain.tex`, `explain.pdf` | full explanation of the package (features, replay, null-space update, experiments) |

## Run the code

Requires Python 3 with NumPy, SciPy and scikit-learn.

```bash
python stable_continual_learning.py --check   # numerical checks (seconds)
python stable_continual_learning.py           # demo: three methods on one split
python benchmark.py                           # about 10 minutes on one CPU core
```

To also check the step against the PR, make `polynomial_cl` importable before `--check`, e.g.
`PYTHONPATH=/path/to/polynomial_cl_core-diagonal_control/src python stable_continual_learning.py --check`.
Without it the parity check prints "skipped".

## Build the PDFs

Needs a TeX distribution with `pdflatex` and the packages `lmodern`, `microtype`, `amsmath`,
`amsthm`, `mathtools`, `enumitem`, `booktabs`, `hyperref` (all in TeX Live and MiKTeX). Run each
file twice so cross-references resolve:

```bash
pdflatex math.tex && pdflatex math.tex
pdflatex explain.tex && pdflatex explain.tex
```

or `latexmk -pdf math.tex explain.tex`. Delete the `.aux`, `.log` and `.out` files afterwards if
you do not want them in the repository.

## Keeping the documents in sync

The numbers in section 5 of `math.tex` come from `python benchmark.py`; if the code or the
protocol changes, rerun the benchmark and update that section and then rebuild the PDFs.
