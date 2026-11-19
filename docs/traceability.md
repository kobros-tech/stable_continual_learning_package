# Traceability: claim, proof, test, experiment

Three evidence levels are tracked separately and are not interchangeable:

- **Proven**: a Lean theorem, from explicit assumptions, about the mathematical update.
- **Tested**: a pytest checks that the numerical implementation satisfies the assumptions or the
  conclusion up to floating-point tolerance. It does not prove them.
- **Observed**: an experiment in `benchmark.py` supports the claim on one dataset under one
  protocol. It proves nothing.

Lean status: "compiled" means `lake build` and the axiom audit passed in CI on PR #3. It means
Lean accepted the statement and proof. That the statement expresses the intended mathematics is a
separate question, answered by the mapping in this document and by review against `math.tex`.
The `formal` workflow runs Lean only; pytest results are separate evidence.

## Assumptions the Lean theorems take from the implementation

| ID  | Assumption                                                          | Used by           | Checked by (pytest)                                                                                                                                             |
| --- | ------------------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A0  | Rates are nonnegative: `η_g ≥ 0` and `m_g ≥ 0`, so `d_j ≥ 0`; the implementation additionally restricts `m_g ≤ 1` | M1-M3 do not need it; M4, M5 (planned) do | `test_invalid_control_factors_are_rejected` (`m`), `test_negative_base_rates_are_rejected` (`η`), `test_incompatible_null_basis_inputs_are_rejected` (negative `d`) |
| A1  | Old features are a prefix of the expanded feature vector; old features keep their degree | M1, M2 (prefix)   | `test_lower_degree_features_are_a_prefix_of_higher_degree_features`, `test_features_are_ordered_by_degree`, `test_rates_are_prefix_stable`                       |
| A2  | `A N = 0` for `A = R D^{1/2}` (exact null basis)                    | M3                | `test_null_basis_is_orthonormal_and_annihilated_by_A` (residual below `1e-9`, so the guarantee holds up to that tolerance, not exactly)                          |
| A3  | `N^T N = I`                                                         | M4, M5 (planned)  | same test                                                                                                                                                       |
| A4  | `‖A N‖₂ ≤ τ` for the relaxed basis                                  | M4 (planned)      | `test_relaxed_logit_change_respects_the_spectral_bound` (indirect: it checks the consequence)                                                                    |

## Claims

| Claim                                                  | `math.tex`      | Proven (Lean)                                                  | Tested (pytest)                                                                                        | Observed |
| ------------------------------------------------------ | --------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------- |
| Zero-padded expansion preserves logits                 | §2              | M1 `expand_logits`, `expand_logits_matrix` (compiled)          | `test_expansion_preserves_logits_on_real_features`, `test_expansion_appends_zero_coefficients_and_keeps_old_ones` | n/a      |
| `d_j = 0` implies row `j` of `ΔW` is zero              | Prop 1(b)       | M2 `nullStep_row_eq_zero`, `sgdStep_row_eq_zero` (compiled)     | `test_frozen_rows_do_not_move`, `test_zero_factor_freezes_a_degree`                                    | n/a      |
| `m_g = 0` freezes degree `g`                           | §2              | M2 `nullStep_frozen_of_control_zero` (compiled)                 | `test_zero_factor_freezes_a_degree`                                                                    | n/a      |
| Rates are prefix-stable under expansion (needs A1)     | §2              | M2 `rate_prefix_stable` (compiled)                              | `test_rates_are_prefix_stable`                                                                         | n/a      |
| Exact step preserves reference logits (needs A2)       | Prop 1(a)       | M3 `ref_preserved_exact`, `ref_preserved_run` (compiled)       | `test_exact_step_leaves_reference_logits_unchanged`                                                    | n/a      |
| Strict step is non-ascent, `⟨G,ΔW⟩ = -‖Pu‖²`           | Prop 1(d)       | M4, not started                                                | `test_step_is_a_descent_direction_with_the_stated_value`                                               | n/a      |
| Relaxed residual: `‖RΔW + βAu‖_F ≤ (1-β)τ‖u‖_F` (needs A3, A4) | Prop 2(c) | M4, not started                                                | `test_relaxed_logit_change_respects_the_spectral_bound`                                                | n/a      |
| Strict step is the unique constrained minimiser        | Prop 1(c)       | M5, not started                                                | `test_exact_step_is_the_constrained_minimiser`                                                         | n/a      |
| Strict step lowers forgetting                          | §5              | not a theorem                                                  | n/a                                                                                                    | supported on `digits`; also lowers newest-task accuracy |
| Relaxed variant improves the accuracy/forgetting trade-off | §5          | not a theorem                                                  | n/a                                                                                                    | meets the accuracy bound at memory 100, misses it at 60 |
| Degree control lowers forgetting                       | §5              | not a theorem                                                  | n/a                                                                                                    | no evidence under the stratified protocol |

## Notes

- The relaxed bound subtracts `β A u` (`= β R D G`). `‖RΔW‖` alone is **not** bounded by
  `(1-β) τ ‖u‖`: on 300 random cases it failed 258 times. The existing test already uses the
  correct quantity.
- `math.tex` Prop 2 uses `τ = tol · max(σ₁, 1)`, while the test above uses `tol · σ₁`. They
  agree whenever `σ₁ ≥ 1`, which holds for its random fixture; aligning the test with the
  spec is a small follow-up.
- Exact reference preservation (M3) is stated for the exact null basis. `null_basis` is a
  numerical SVD with a tolerance, so for the implementation the guarantee is "up to A2's
  residual", which is a numerical statement, not a theorem.
- Nonnegative rates (A0). In Lean, `Real.sqrt` sends a negative number to `0`, so `sqrtDiag` is
  defined for every real `d`, while the Python code rejects negative rates. The M1-M3 theorems
  hold for all real `d` and do not use `d ≥ 0`. `rate_nonneg` records that `η_g ≥ 0` and
  `m_g ≥ 0` give `d_j ≥ 0`, and `sqrtDiag_mul_self` records that `D^{1/2} D^{1/2} = D` needs
  it; both compiled in CI on PR #3 and are in the axiom audit. Theorems that use
  `D = D^{1/2} D^{1/2}` (planned M4, M5) must take `∀ j, 0 ≤ d j` as an explicit hypothesis.
