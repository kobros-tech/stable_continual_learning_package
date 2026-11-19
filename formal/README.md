# Formal verification (Lean 4 + Mathlib)

Lean proves properties of the **mathematical** update specified in `../math.tex`. It does not
verify the Python code, floating-point behaviour, or any claim about accuracy or forgetting.
The Python package never imports Lean; Lean is a development dependency only.

## Status

M0-M3 **built in CI on PR #3** and M4-M5 **built in CI on PR #5**: `lake build` succeeded and the
axiom audit in `Audit.lean` passed, with Lean and Mathlib pinned to `v4.36.0` and the manifest
committed. The PR #3 build includes
`rate_nonneg` and `sqrtDiag_mul_self` (`Definitions.lean`), which make the nonnegative-rate domain
explicit. The `formal` workflow runs Lean only; the Python tests run in the separate `python` workflow
(pytest and pre-commit) and are separate evidence.

Two different checks are involved, and a green build gives only the first:

1. **Lean accepts the statements and proofs.** Established by CI.
2. **The statements faithfully express the intended mathematics.** Established by review against
   `math.tex`, recorded per claim in `../docs/traceability.md`. A green build does not show this.

A theorem being compiled is not the same as the Python implementation satisfying it: the
numerical `null_basis` is tested only up to a residual, not exactly (see the traceability
document).

| Milestone | File                         | Result                                                     |
| --------- | ---------------------------- | ---------------------------------------------------------- |
| M0        | `lakefile.toml`, CI          | Lean and Mathlib pinned together; build; proof-integrity   |
| M1        | `StableCL/Expansion.lean`    | `[W; 0]` expansion preserves logits                        |
| M2        | `StableCL/DegreeControl.lean`| zero effective rate freezes a row; prefix stability        |
| M3        | `StableCL/RefPreservation.lean` | `A N = 0` implies `R ΔW = 0`, also over a whole experience |
| M4a       | `StableCL/Descent.lean`      | descent identity, non-ascent, strict descent, stall iff, `β = 1` is SGD         |
| M4b       | `StableCL/Relaxed.lean`      | `‖R ΔW + β R D G‖_F ≤ (1-β) τ ‖D^{1/2} G‖_F`             |
| M5        | `StableCL/Optimality.lean`   | strict step is the unique constrained minimiser         |

**M4 and M5 are compiled** (`Frobenius`, `Descent`, `Relaxed`, `Optimality`) and every
declaration in them is in the axiom audit. Compilation says Lean accepted the statements and
proofs; that they express `math.tex` is the review recorded in `../docs/traceability.md`.
Assumptions M4/M5 add beyond M1-M3 are explicit hypotheses: `d ≥ 0` (A0), `NᵀN = I` (A3), the
spectral bound (A4, M4b) and completeness of `N` (A5, M5). M6 (numerical checks, pytest) and M7
(benchmarks) are not Lean work and are not exercised by the `formal` workflow.

## Building locally

`lake-manifest.json` is committed and pins Mathlib and its dependencies to exact commits.

```sh
cd formal
lake exe cache get   # prebuilt Mathlib oleans; skip and the first build takes hours
lake build
```

`lean-toolchain` must equal Mathlib's own `lean-toolchain` at the revision in `lakefile.toml`.
To upgrade, change both together, run `lake update`, and commit the new manifest.

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
