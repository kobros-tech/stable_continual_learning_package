# Formal verification (Lean 4 + Mathlib)

Lean proves properties of the **mathematical** update specified in `../math.tex`. It does not
verify the Python code, floating-point behaviour, or any claim about accuracy or forgetting.
The Python package never imports Lean; Lean is a development dependency only.

## Status

**Every `.lean` file here is a DRAFT: it has not been compiled.** The drafts were written
without a Lean toolchain or Mathlib build cache available, against Mathlib `v4.36.0` (lemma
names were checked in that tag's source, but nothing was elaborated). Treat a statement as
proven only when CI is green and `docs/traceability.md` says so.

| Milestone | File                         | Result                                                     |
| --------- | ---------------------------- | ---------------------------------------------------------- |
| M0        | `lakefile.toml`, CI          | Lean and Mathlib pinned together; build; proof-integrity   |
| M1        | `StableCL/Expansion.lean`    | `[W; 0]` expansion preserves logits                        |
| M2        | `StableCL/DegreeControl.lean`| zero effective rate freezes a row; prefix stability        |
| M3        | `StableCL/RefPreservation.lean` | `A N = 0` implies `R ΔW = 0`, also over a whole experience |

M4 (descent, relaxed residual bound) and M5 (constrained minimiser) are not started. They begin
only after M0-M3 compile and their statements have been checked against `math.tex`.

## First-time setup

```sh
cd formal
lake update          # resolves Mathlib v4.36.0 and writes lake-manifest.json: COMMIT IT
lake exe cache get   # prebuilt Mathlib oleans; skip and the first build takes hours
lake build
```

`lean-toolchain` must equal Mathlib's own `lean-toolchain` at the revision in `lakefile.toml`.
To upgrade, change both together, run `lake update`, commit the new manifest.

## Proof integrity

Searching for `sorry` is not enough. `StableCL/Audit.lean` defines `#assert_standard_axioms`,
which fails the build if a listed theorem depends on any axiom other than `propext`,
`Classical.choice`, `Quot.sound`. That covers `sorryAx` (however introduced), `native_decide`
and hand-written `axiom`s. `scripts/check_audit_coverage.sh` fails if a theorem is not listed in
the audit.

## Rules

1. Each theorem maps to a labelled proposition in `math.tex` (see `../docs/traceability.md`).
2. Each assumption a theorem takes from the implementation is a named hypothesis, mapped to a
   pytest.
3. Proven, numerically tested, and empirically observed are different evidence levels and are
   never merged in one status.
