-- STATUS: the M0-M3 entries compiled in CI on PR #3 and pass the axiom audit. The entries from
-- Frobenius, Descent, Relaxed and Optimality (M4, M5) are DRAFT, not yet compiled.
-- Copyright (c) 2026 Mohamed Alkobrosli
-- SPDX-License-Identifier: MIT
import Lean
import StableCL.Expansion
import StableCL.DegreeControl
import StableCL.RefPreservation
import StableCL.Frobenius
import StableCL.Descent
import StableCL.Relaxed
import StableCL.Optimality

/-!
# Proof-integrity audit

A text search for `sorry` is not enough (`admit`, macros, `native_decide`, new `axiom`s).
`#assert_standard_axioms` fails the build if a theorem depends on any axiom other than
`propext`, `Classical.choice`, `Quot.sound`; this catches `sorryAx` however it got in.

`formal/scripts/check_audit_coverage.sh` checks that every theorem is listed below.
-/

open Lean Elab Command

elab "#assert_standard_axioms " ids:ident+ : command => do
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  for id in ids do
    let n ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo id
    let axs ← collectAxioms n
    for a in axs do
      unless allowed.contains a do
        throwError "{n} depends on non-standard axiom {a}"

#assert_standard_axioms
  StableCL.rate_nonneg
  StableCL.sqrtDiag_mul_self
  StableCL.expand_logits
  StableCL.expand_logits_matrix
  StableCL.expand_old_rows
  StableCL.expand_new_rows
  StableCL.sgdStep_row_eq_zero
  StableCL.nullStep_row_eq_zero
  StableCL.rate_eq_zero_of_control_zero
  StableCL.nullStep_frozen_of_control_zero
  StableCL.rate_prefix_stable
  StableCL.rate_new_feature
  StableCL.ref_preserved_exact
  StableCL.ref_preserved_run
  StableCL.frob_apply
  StableCL.frob_comm
  StableCL.frob_zero_right
  StableCL.frob_add_right
  StableCL.frob_sub_right
  StableCL.frob_neg_right
  StableCL.frob_smul_right
  StableCL.frob_add_left
  StableCL.frob_sub_left
  StableCL.frob_neg_left
  StableCL.frob_smul_left
  StableCL.frob_neg_neg
  StableCL.frob_smul_smul
  StableCL.frob_self_nonneg
  StableCL.frob_self_eq_zero
  StableCL.frob_symm_mul_left
  StableCL.sqrtDiag_transpose
  StableCL.frob_sqrtDiag_right
  StableCL.proj_transpose
  StableCL.proj_idem
  StableCL.frob_proj
  StableCL.frob_proj_le
  StableCL.frob_transpose_le
  StableCL.nullStep_inner
  StableCL.deltaW_inner
  StableCL.deltaW_inner_nonpos
  StableCL.deltaW_inner_neg
  StableCL.nullStep_zero_eq_zero_iff
  StableCL.nullStep_one
  StableCL.ref_change_decomp
  StableCL.ref_residual_eq
  StableCL.ref_residual_sq
  StableCL.ref_residual_norm
  StableCL.isqrt_mul_apply
  StableCL.sqrtDiag_mul_apply
  StableCL.sqrtDiag_isqrt_mul
  StableCL.isqrt_sqrtDiag_apply
  StableCL.penalty_eq
  StableCL.objective_eq
  StableCL.frob_eq_of_agree_on_support
  StableCL.complete_square
  StableCL.nullStep_zero_eq
  StableCL.strictStep_feasible
  StableCL.isqrt_mem_ker
  StableCL.objective_expand
  StableCL.strictStep_unique_minimiser
