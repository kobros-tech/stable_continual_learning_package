# Traceability: claim, proof, test, experiment

Three evidence levels are tracked separately and are not interchangeable:

- **Proven**: a Lean theorem, from explicit assumptions, about the mathematical update.
- **Tested**: a pytest checks that the numerical implementation satisfies the assumptions or the
  conclusion up to floating-point tolerance. It does not prove them.
- **Observed**: an experiment supports the claim on one dataset under one protocol. It proves
  nothing. This repository no longer contains experiment code; the Observed entries below are
  historical results of the removed benchmark, preserved at the Git tag `legacy-m7-benchmark`. They
  are not regenerated here and are not evidence for any later API or paper.

## Names: mathematical object, Lean, Python

Proof evidence and numerical evidence are kept apart. A Lean declaration is never cited as evidence
that a Python function is correct, and a pytest is never cited as proof of a theorem. Lean proves
statements about the mathematical update from explicit assumptions; Python implements the numerical
procedure; pytest checks that procedure. The first four rows below are objects the theorems talk
about, the last two are procedures the theorems assume.

| Mathematical object                         | Lean (proved, under assumptions)          | Python (implementation)  | Numerical evidence                           |
| ------------------------------------------- | ----------------------------------------- | ------------------------ | -------------------------------------------- |
| `d_j = η_g m_g`, `D = diag(d)`              | `rate` (`G = 4`, `Fin 4`)                 | `diagonal_rates`         | `tests/test_degree_control.py`               |
| Ordinary step `D G`                         | `sgdStep`                                 | `ordinary_step`          | `tests/test_steps.py` (equals `β = 1`)       |
| Relaxed step `D^{1/2}[P + β(I-P)]D^{1/2}G`  | `nullStep`                                | `relaxed_step`           | `tests/test_null_space.py`, `test_assumptions.py` |
| `β = 1`, `β = 0`                            | `nullStep_one`, M3 `ref_preserved_exact`  | `relaxed_step(beta=...)` | same tests                                   |
| Null-space basis `N`, `P = N Nᵀ`            | hypotheses A2-A5, never constructed       | `null_basis`             | A2-A5 tests (up to tolerance)                |
| Achieved `‖A N‖₂` (the `τ` of A4)           | hypothesis A4                             | `spectral_bound`         | `tests/test_steps.py`                        |
| Free dimension `r`                          | none (Prop 1(e))                          | `free_dim`               | pytest only                                  |
| Zero-padding expansion `W ↦ [W; 0]`         | M1 (`Matrix.fromRows W 0`)                | none; numpy in the test  | `tests/test_expansion_logits.py`             |

Lean keeps its own names (`nullStep`, `sgdStep`); the Python names are `relaxed_step` and
`ordinary_step`. Python's `null_basis` is a numerical SVD routine, so the rows "uniqueness of the
exact step" (M5) and "`N` is a null basis" are different claims with different evidence: M5 is a
theorem that assumes A5, and A5 for the computed basis is checked by pytest only.

Group counts. Python accepts any number `G ≥ 1` of rate groups. Lean states `rate` and the M2
theorems about degree groups for `Fin 4` (`G = 4`) only. M3-M5 take an arbitrary `d : p → ℝ` and do
not depend on `G`. For `G ≠ 4` the construction of `d` and the freezing and prefix claims are
covered by pytest only (`test_group_count_is_inferred_from_the_base_rates`). Generalising `Fin 4`
in Lean is deferred and is not implied by this package.

Lean status: "compiled" means `lake build` and the axiom audit passed in CI (M0-M3 on PR #3, M4-M5 on
PR #5). It means
Lean accepted the statement and proof. That the statement expresses the intended mathematics is a
separate question, answered by the mapping in this document and by review against `math.tex`.
The `formal` workflow runs Lean only; pytest runs in the separate `python` workflow, and its
results are separate evidence.

## Assumptions the Lean theorems take from the implementation

| ID  | Assumption                                                          | Used by           | Checked by (pytest)                                                                                                                                             |
| --- | ------------------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A0  | Rates are nonnegative: `η_g ≥ 0` and `m_g ≥ 0`, so `d_j ≥ 0`; the implementation additionally restricts `m_g ≤ 1` | M1-M3 do not need it; M4, M5 do | `test_invalid_control_factors_are_rejected` (`m`), `test_negative_base_rates_are_rejected` (`η`), `test_incompatible_null_basis_inputs_are_rejected` (negative `d`) |
| A1  | Old features are a prefix of the expanded feature vector; old features keep their degree | M1, M2 (prefix)   | Not checked in this repository: the feature map lives in `polynomial_cl_core`, whose tests must check that old features are a prefix of the new ones and keep their degree (cross-repo; not run by this repository's CI). Here only `test_rates_are_prefix_stable` checks that `diagonal_rates` is prefix-stable for a given group vector |
| A2  | `A N = 0` for `A = R D^{1/2}` (exact null basis)                    | M3                | `test_null_basis_is_orthonormal_and_annihilated_by_A` (residual below `1e-9`, so the guarantee holds up to that tolerance, not exactly)                          |
| A3  | `N^T N = I`                                                         | M4                | same test                                                                                                                                                       |
| A4  | `‖A N‖₂ ≤ τ` for the relaxed basis, stated in Lean as `‖(A N) X‖_F ≤ τ ‖X‖_F` for all `X` | M4b           | `test_a4_spectral_norm_of_A_N_is_at_most_tau` (direct: `‖A N‖₂ ≤ τ`, 4 tolerances x 48 cases), `test_cutoff_consistency_links_the_svd_threshold_to_a4`, and indirectly `test_relaxed_logit_change_respects_the_spectral_bound` |
| A5  | `N` spans all of `null(A)`: `N N^T V = V` whenever `A V = 0` (the `tol → 0` hypothesis of Prop 1) | M5           | `test_exact_null_basis_spans_the_whole_null_space`, `test_exact_null_basis_is_complete_for_rank_deficient_references`, `test_a5_projector_fixes_every_vector_of_an_independently_built_null_space` (N Nᵀ V = V for V from a QR-built null space), `test_a5_holds_for_nearly_degenerate_singular_values` (dimension equals `p - rank A`, with the rank taken under `null_basis`'s own threshold `σ > tol·max(σ₁,1)`; given A2 and orthonormal columns (A3) this is equivalent to `N` spanning `null(A)`, which is the form `hcomp` states in Lean) |

## Claims

| Claim                                                  | `math.tex`      | Proven (Lean)                                                  | Tested (pytest)                                                                                        | Observed (historical) |
| ------------------------------------------------------ | --------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------- |
| Zero-padded expansion preserves logits                 | §2              | M1 `expand_logits`, `expand_logits_matrix` (compiled)          | `test_zero_padding_preserves_logits`, `test_zero_padding_keeps_old_coefficients_and_starts_new_ones_at_zero` (plain numpy, no feature map) | n/a      |
| `d_j = 0` implies row `j` of `ΔW` is zero              | Prop 1(b)       | M2 `nullStep_row_eq_zero`, `sgdStep_row_eq_zero` (compiled)     | `test_frozen_rows_do_not_move`, `test_zero_factor_freezes_a_degree`                                    | n/a      |
| `m_g = 0` freezes degree `g`                           | §2              | M2 `nullStep_frozen_of_control_zero` (compiled)                 | `test_zero_factor_freezes_a_degree`                                                                    | n/a      |
| Rates are prefix-stable under expansion (needs A1)     | §2              | M2 `rate_prefix_stable` (compiled)                              | `test_rates_are_prefix_stable`                                                                         | n/a      |
| Exact step preserves reference logits (needs A2)       | Prop 1(a)       | M3 `ref_preserved_exact`, `ref_preserved_run` (compiled)       | `test_exact_step_leaves_reference_logits_unchanged`                                                    | n/a      |
| Strict step is non-ascent, `⟨G,ΔW⟩ = -‖Pu‖²`; stalls iff `Pu = 0` (needs A3) | Prop 1(d) | M4a `nullStep_inner`, `deltaW_inner`, `deltaW_inner_nonpos`, `nullStep_zero_eq_zero_iff` (compiled, PR #5) | `test_step_is_a_descent_direction_with_the_stated_value`                                               | n/a      |
| Relaxed step is non-ascent, strictly descending if `β > 0`, `u ≠ 0`; `β = 1` is rate-weighted SGD | Prop 2(a), 2(b) | M4a `deltaW_inner_nonpos`, `deltaW_inner_neg`, `nullStep_one` (compiled, PR #5) | `test_beta_one_is_ordinary_rate_weighted_sgd`, `test_step_is_a_descent_direction_with_the_stated_value` | n/a |
| Relaxed residual: `‖RΔW + βAu‖_F ≤ (1-β)τ‖u‖_F` (needs A0, A3, A4) | Prop 2(c) | M4b `ref_change_decomp`, `ref_residual_eq`, `ref_residual_sq`, `ref_residual_norm` (compiled, PR #5) | `test_relaxed_logit_change_respects_the_spectral_bound`                                                | n/a      |
| Strict step is the unique constrained minimiser over the feasible set (needs A0, A2, A5) | Prop 1(c) | M5 `strictStep_unique_minimiser`, `objective_expand` (compiled, PR #5) | `test_exact_step_is_the_constrained_minimiser`                                                         | n/a      |
| Strict step approximately preserves reference logits: `‖RΔW‖_F ≤ τ‖D^{1/2}G‖_F` (needs A3, A4) | Prop 2(c) at `β = 0` | M4b `ref_residual_norm` with `β = 0` (instance of the compiled theorem; no separate Lean declaration) | `test_strict_step_approximate_preservation_is_the_beta_zero_case`, `test_relaxed_residual_bound_over_a_beta_grid` | measured drift not recorded |
| Strict step lowers forgetting                          | §5              | not a theorem                                                  | n/a                                                                                                    | supported on `digits`; also lowers newest-task accuracy |
| Relaxed variant improves the accuracy/forgetting trade-off | §5          | not a theorem                                                  | n/a                                                                                                    | meets the accuracy bound at memory 100, misses it at 60 |
| Degree control lowers forgetting                       | §5              | not a theorem                                                  | n/a                                                                                                    | no evidence under the stratified protocol |

## Notes

- The relaxed bound subtracts `β A u` (`= β R D G`). `‖RΔW‖` alone is **not** bounded by
  `(1-β) τ ‖u‖`: on 300 random cases it failed 258 times. The existing test already uses the
  correct quantity.
- `math.tex` Prop 2 and `test_relaxed_logit_change_respects_the_spectral_bound` both use
  `τ = tol · max(σ₁, 1)`. The tests scale `R` so that `σ₁` is far below 1 (about 0.1) and far above
  1 (about 10 to 3500), and `test_relaxed_residual_bound_with_sigma1_around_one` places `σ₁` at
  0.5, 1 and 2, so both sides of the `max(·, 1)` branch point are exercised, not only
  `σ₁ ≥ 1`.
- Exact reference preservation (M3) is stated for the exact null basis. `null_basis` is a
  numerical SVD with a tolerance, so for the implementation the guarantee is "up to A2's
  residual", which is a numerical statement, not a theorem.
- Nonnegative rates (A0). In Lean, `Real.sqrt` sends a negative number to `0`, so `sqrtDiag` is
  defined for every real `d`, while the Python code rejects negative rates. The M1-M3 theorems
  hold for all real `d` and do not use `d ≥ 0`. `rate_nonneg` records that `η_g ≥ 0` and
  `m_g ≥ 0` give `d_j ≥ 0`, and `sqrtDiag_mul_self` records that `D^{1/2} D^{1/2} = D` needs
  it; both compiled in CI on PR #3 and are in the axiom audit. Theorems that use
  `D = D^{1/2} D^{1/2}` (M4, M5) take `∀ j, 0 ≤ d j` as an explicit hypothesis.
- M4 and M5 compiled in CI on PR #5 and are in the axiom audit; their rows say so. They follow
  `math.tex` Prop 1(c), 1(d), 2(a), 2(b), 2(c). Not covered by any Lean theorem: Prop 1(e) (rank of the free space; pytest only)
  and Prop 1(f) (correspondence with the earlier `polynomial_cl` implementation). Prop 1(f) had a parity
  test that was removed together with the experiment code; it is currently not tested in this
  repository, and the equivalence with `polynomial_cl_core` is to be established by core's tests.
- Spectral bound (A4). Lean takes `‖A N‖₂ ≤ τ` as a hypothesis in its defining form. That the
  SVD threshold gives `‖A N‖₂ = σ_{k+1} ≤ τ` is not formalised; it is checked numerically.
- Completeness (A5). Exact reference preservation (M3) needs only `A N = 0`. Constrained
  optimality (M5) additionally needs that `N` is not a proper subspace of `null(A)`; with a
  smaller basis the step is feasible but not the minimiser.
  The completeness tests count rank with the same threshold as `null_basis`
  (`σ > tol·max(σ₁,1)`), not `np.linalg.matrix_rank`'s default `σ₁·max(shape)·eps`; the two
  can disagree on near-null singular values. Numerical completeness is a floating-point check
  and is not implied by any Lean theorem.
- Uniqueness in M5 is over the feasible set `{Z : Z_j = 0 for d_j = 0, R Z = 0}`, not over all
  of `R^{p x K}`; the objective does not penalise rows outside `S`.
- M6 (`tests/test_assumptions.py`) checks A2-A5 and the approximate guarantees over 48 random,
  rank-deficient, partly frozen and rescaled cases (`σ₁` from about 0.07 to about 3500). These are
  floating-point tests. They do not prove that `np.linalg.svd` satisfies A2-A5 in every case,
  and Lean does not verify them. What would prove them is a Lean theorem about the exact
  real-arithmetic SVD (right singular vectors with `σ ≤ τ` satisfy `NᵀN = I`, `‖A N‖₂ ≤ τ`, and,
  at `τ = 0`, `A N = 0` and completeness). That theorem is **not** in this package; floating-point
  behaviour of the SVD routine could not be covered by it in any case.
- M7 (the benchmark) is historical. Its results were preliminary, `digits`-only observations from
  code that has been removed from this repository; the source, protocol and raw-result handling are
  preserved at the Git tag `legacy-m7-benchmark`. The Observed entries above are `digits` only and
  are not regenerated here.
- The `descent` results concern the linearised loss (directional derivative) only. They do not
  imply that a finite step decreases the nonlinear loss.
