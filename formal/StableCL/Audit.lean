-- STATUS: DRAFT, not yet compiled.
-- Copyright (c) 2026 Mohamed Alkobrosli
-- SPDX-License-Identifier: MIT
import Lean
import StableCL.Expansion
import StableCL.DegreeControl
import StableCL.RefPreservation

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
