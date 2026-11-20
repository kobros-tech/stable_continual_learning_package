-- STATUS: compiled in CI on PR #3 (M0-M3 build + axiom audit); unchanged since.
-- Copyright (c) 2026 Mohamed Alkobrosli
-- SPDX-License-Identifier: MIT
import StableCL.Definitions

/-!
# M3. Exact reference preservation   (`math.tex` Prop. 1(a))

With `A = R D^{1/2}` and `β = 0`:  `R ΔW = -(A N)(Nᵀ D^{1/2} G)`, so `A N = 0` alone gives
`R ΔW = 0`. Orthonormality `NᵀN = I` is **not** used; it is needed only for descent and
optimality (later milestones). The relaxed case `β > 0` is deliberately not treated here.

Rows of `R` are the reference feature vectors, so `R * W` is the matrix of reference logits.
The statement is over all `K` columns, which contains the "seen columns" of `math.tex`.
-/

open Matrix

namespace StableCL

variable {m p q K : Type*} [Fintype p] [Fintype q] [DecidableEq p]

/-- One exact step leaves the reference logits unchanged. Hypothesis: `A N = 0`. -/
theorem ref_preserved_exact (R : Matrix m p ℝ) (d : p → ℝ) (N : Matrix p q ℝ)
    (G : Matrix p K ℝ) (hAN : R * sqrtDiag d * N = 0) :
    R * nullStep d N G 0 = 0 := by
  have key : R * nullStep d N G 0 = (R * sqrtDiag d * N) * (Nᵀ * (sqrtDiag d * G)) := by
    simp [nullStep, Matrix.mul_assoc]
  rw [key, hAN, Matrix.zero_mul]

/-- Apply the exact step `W ↦ W - nullStep d N G 0` for each gradient in the list.
`d` and `N` (hence `R`'s constraint `A N = 0`) are fixed for the whole run, as in one experience
of `train_task`. -/
noncomputable def run (d : p → ℝ) (N : Matrix p q ℝ) :
    List (Matrix p K ℝ) → Matrix p K ℝ → Matrix p K ℝ
  | [], W => W
  | G :: Gs, W => run d N Gs (W - nullStep d N G 0)

/-- Reference logits are unchanged after a whole experience, for fixed `R`, `D`, `N`. -/
theorem ref_preserved_run (R : Matrix m p ℝ) (d : p → ℝ) (N : Matrix p q ℝ)
    (hAN : R * sqrtDiag d * N = 0) :
    ∀ (Gs : List (Matrix p K ℝ)) (W : Matrix p K ℝ), R * run d N Gs W = R * W
  | [], W => rfl
  | G :: Gs, W => by
    show R * run d N Gs (W - nullStep d N G 0) = R * W
    rw [ref_preserved_run R d N hAN Gs, Matrix.mul_sub, ref_preserved_exact R d N G hAN, sub_zero]

end StableCL
