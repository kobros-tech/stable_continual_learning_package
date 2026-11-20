-- STATUS: compiled in CI on PR #3 (M0-M3 build + axiom audit); unchanged since.
-- Copyright (c) 2026 Mohamed Alkobrosli
-- SPDX-License-Identifier: MIT
import StableCL.Definitions

/-!
# M1. Zero-padding expansion preserves logits   (`math.tex` §2; `stable_cl.features.expand`)

`W' = [W; 0]`, `φ' = [φ; ψ]`  ⟹  `φ'ᵀ W' = φᵀ W`.

Old and new features are indexed by `p ⊕ q`: old features are `Sum.inl`, appended ones `Sum.inr`.
**Nothing here refers to how monomials are generated.** That the real feature map has this
prefix structure is assumption A1 in `docs/traceability.md`, checked by pytest, not by Lean.
-/

open Matrix

namespace StableCL

variable {p q K n : Type*} [Fintype p] [Fintype q]

/-- Single input: logits after expansion equal logits before. -/
theorem expand_logits (W : Matrix p K ℝ) (φ : p → ℝ) (ψ : q → ℝ) :
    Sum.elim φ ψ ᵥ* Matrix.fromRows W (0 : Matrix q K ℝ) = φ ᵥ* W := by
  rw [Matrix.sumElim_vecMul_fromRows]
  simp

/-- A batch of inputs (rows of `Φ`, `Ψ`). -/
theorem expand_logits_matrix (Φ : Matrix n p ℝ) (Ψ : Matrix n q ℝ) (W : Matrix p K ℝ) :
    Matrix.fromCols Φ Ψ * Matrix.fromRows W (0 : Matrix q K ℝ) = Φ * W := by
  rw [Matrix.fromCols_mul_fromRows]
  simp

/-- Old coefficients are kept. -/
theorem expand_old_rows (W : Matrix p K ℝ) (i : p) (k : K) :
    Matrix.fromRows W (0 : Matrix q K ℝ) (Sum.inl i) k = W i k := by
  simp [Matrix.fromRows_apply_inl]

/-- New coefficients start at zero. -/
theorem expand_new_rows (W : Matrix p K ℝ) (i : q) (k : K) :
    Matrix.fromRows W (0 : Matrix q K ℝ) (Sum.inr i) k = 0 := by
  simp [Matrix.fromRows_apply_inr]

end StableCL
