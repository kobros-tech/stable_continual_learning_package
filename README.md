# Stable continual learning package

The mathematical update for stable continual learning: a diagonal-rate null-space step, its numpy
implementation, and its Lean 4 formalisation. This repository does not contain an experiment
framework or benchmarks.

The update, with `D = diag(d)`, `A = R D^{1/2}`, `P = N Nᵀ` (`N` an orthonormal basis of a numerical
null space of `A`) and relaxation `β`, is

```text
ΔW = -D^{1/2} [P + β (I - P)] D^{1/2} G
```

`β = 0` is the exact step (reference-preserving when `A N = 0`), `β = 1` is rate-weighted SGD, and
`0 < β < 1` is the relaxed step. The relaxed step is not unconditionally reference-preserving; its
residual bound holds under the assumptions stated in `math.tex` and `formal/`.

## Layout

| path                        | role                                                                                   |
| --------------------------- | -------------------------------------------------------------------------------------- |
| `stable_cl/optimization.py` | `diagonal_rates`, `null_basis`, `ordinary_step`, `relaxed_step`, `spectral_bound`, `free_dim` |
| `tests/`                    | pytest suite: numerical properties of every function; `test_assumptions.py` checks A2-A5 |
| `formal/`                   | Lean 4 + Mathlib proofs about the exact update (M0-M5); see `formal/README.md`         |
| `docs/traceability.md`      | claim, Lean statement, numerical test: kept as separate evidence                       |
| `math.tex`                  | specification: definitions, propositions, proofs                                       |
| `explain.tex`               | legacy explanation, written for the package before the experiment code was removed   |

Every function has a docstring with its purpose, argument shapes, return value and, where it
implements a mathematical operation, the equation (Sphinx `.. math::` syntax), as in `math.tex`.

## Public API

Effective rates `d` (learning rate included) are the only rate input; no step applies another rate.

| function                              | meaning                                                              |
| ------------------------------------- | -------------------------------------------------------------------- |
| `diagonal_rates(base_rates, group, control=None)` | `d_j = η_g m_g`; the group count is `len(base_rates)`, no scalar broadcast |
| `null_basis(R, d, tol)`               | orthonormal numerical null space `N` of `A = R D^{1/2}`              |
| `ordinary_step(G, d)`                 | `D G` (unsigned; `ΔW = -step`)                                       |
| `relaxed_step(G, N, d, beta)`         | `D^{1/2}[P + β(I − P)]D^{1/2}G` (unsigned)                           |
| `spectral_bound(R, d, N)`             | the achieved `‖A N‖₂`                                                |

## Three kinds of evidence

- **Proven**: a Lean theorem from explicit assumptions (`formal/`).
- **Tested**: a pytest checks the numerical implementation up to floating-point tolerance. It does
  not prove the assumptions or the conclusion.
- **Observed**: an experiment on data. This repository contains no experiments; the earlier
  observations are historical (see below) and are not evidence for any later API or paper.

## Run the tests

Requires Python 3.11 or newer with NumPy (`pytest` for the tests). `pip install .` installs the
package (numpy only); inside a checkout, `pythonpath = ["."]` in `pyproject.toml` makes `stable_cl`
importable for the tests.

```bash
python -m pytest                  # numerical tests (seconds)
```

## Legacy experiment code

The earlier experiment runner, training loop, benchmark and benchmark workflow, together with their
historical results, were removed from this repository. They are preserved at the Git tag
`legacy-m7-benchmark`. Experiments belong to `polynomial_cl_core`; this repository will not run them.

## Build the PDFs

Needs a TeX distribution with `pdflatex` and the packages `lmodern`, `microtype`, `amsmath`,
`amsthm`, `mathtools`, `enumitem`, `booktabs`, `hyperref` (all in TeX Live and MiKTeX), plus
those loaded by `explain.tex`. Run each file twice so cross-references resolve:

```bash
pdflatex math.tex && pdflatex math.tex
pdflatex explain.tex && pdflatex explain.tex
```

or `latexmk -pdf math.tex explain.tex`. The PDFs are not committed. Delete the `.aux`, `.log` and
`.out` files afterwards if you do not want them in the repository.
