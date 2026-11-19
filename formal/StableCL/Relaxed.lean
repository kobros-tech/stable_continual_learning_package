-- STATUS: DRAFT, not yet compiled.
-- Copyright (c) 2026 Mohamed Alkobrosli
-- SPDX-License-Identifier: MIT
import StableCL.Descent

/-!
# M4b. Relaxed residual bound   (`math.tex` Prop. 2(c))

For `A = R D^{1/2}`, `u = D^{1/2} G`, `ΔW = -nullStep d N G β`:

`R ΔW + β R D G = -(1-β) (A N)(Nᵀ u)`,

hence, if `‖A N‖₂ ≤ τ` and `NᵀN = I`,

`‖R ΔW + β R D G‖_F ≤ (1-β) τ ‖u‖_F`.

This is kept separate from exact preservation (M3, which needs only `A N = 0`). Here **both**
`NᵀN = I` (A3) and `d ≥ 0` (A0) are used. The spectral bound is a hypothesis (A4); how it follows
from the SVD threshold (`‖A N‖₂ = σ_{k+1} ≤ τ`) is not formalised, and is checked numerically.

`‖A N‖₂ ≤ τ` is stated as its defining property `‖(A N) X‖_F ≤ τ ‖X‖_F` for all `X`, in squared
form to avoid square roots; the conclusion is proved in squared form first, then with norms.
-/

open Matrix

namespace StableCL

variable {m p q K : Type*} [Fintype m] [Fintype p] [Fintype q] [Fintype K]
  [DecidableEq p] [DecidableEq q]

/-- `R S = β R D G + (1-β) (A N)(Nᵀ u)`, exactly. -/
theorem ref_change_decomp (R : Matrix m p ℝ) (d : p → ℝ) (hd : ∀ j, 0 ≤ d j)
    (N : Matrix p q ℝ) (G : Matrix p K ℝ) (β : ℝ) :
    R * nullStep d N G β =
      β • (R * Matrix.diagonal d * G)
        + (1 - β) • (R * sqrtDiag d * N * (Nᵀ * (sqrtDiag d * G))) := by
  have e1 : R * (sqrtDiag d * (sqrtDiag d * G)) = R * Matrix.diagonal d * G := by
    rw [← Matrix.mul_assoc (sqrtDiag d), sqrtDiag_mul_self d hd, Matrix.mul_assoc]
  have e2 : R * (sqrtDiag d * (N * Nᵀ * (sqrtDiag d * G)))
      = R * sqrtDiag d * N * (Nᵀ * (sqrtDiag d * G)) := by
    simp only [Matrix.mul_assoc]
  unfold nullStep
  simp only [Matrix.add_mul, Matrix.smul_mul, Matrix.one_mul, Matrix.mul_add, Matrix.mul_smul]
  rw [e1, e2]

/-- The residual `R ΔW + β R D G` equals `-(1-β) (A N)(Nᵀ u)`. -/
theorem ref_residual_eq (R : Matrix m p ℝ) (d : p → ℝ) (hd : ∀ j, 0 ≤ d j)
    (N : Matrix p q ℝ) (G : Matrix p K ℝ) (β : ℝ) :
    R * (-(nullStep d N G β)) + β • (R * Matrix.diagonal d * G)
      = -((1 - β) • (R * sqrtDiag d * N * (Nᵀ * (sqrtDiag d * G)))) := by
  rw [Matrix.mul_neg, ref_change_decomp R d hd N G β]
  abel

/-- Squared form of Prop. 2(c). Hypotheses: `d ≥ 0`, `NᵀN = I`, and `‖A N‖₂ ≤ τ`. -/
theorem ref_residual_sq (R : Matrix m p ℝ) (d : p → ℝ) (hd : ∀ j, 0 ≤ d j)
    (N : Matrix p q ℝ) (G : Matrix p K ℝ) (β τ : ℝ) (hN : Nᵀ * N = 1)
    (hτ : ∀ X : Matrix q K ℝ,
      frob (R * sqrtDiag d * N * X) (R * sqrtDiag d * N * X) ≤ τ ^ 2 * frob X X) :
    frob (R * (-(nullStep d N G β)) + β • (R * Matrix.diagonal d * G))
         (R * (-(nullStep d N G β)) + β • (R * Matrix.diagonal d * G))
      ≤ ((1 - β) * τ) ^ 2 * frob (sqrtDiag d * G) (sqrtDiag d * G) := by
  rw [ref_residual_eq R d hd N G β, frob_neg_neg, frob_smul_smul]
  have h1 := hτ (Nᵀ * (sqrtDiag d * G))
  have h2 := frob_transpose_le N hN (sqrtDiag d * G)
  have h3 := mul_le_mul_of_nonneg_left h2 (sq_nonneg τ)
  have h4 := mul_le_mul_of_nonneg_left (le_trans h1 h3) (sq_nonneg (1 - β))
  exact le_of_le_of_eq h4 (by ring)

/-- Prop. 2(c) with norms `‖X‖_F = √⟨X, X⟩`, for `0 ≤ τ` and `β ≤ 1`. -/
theorem ref_residual_norm (R : Matrix m p ℝ) (d : p → ℝ) (hd : ∀ j, 0 ≤ d j)
    (N : Matrix p q ℝ) (G : Matrix p K ℝ) (β τ : ℝ) (hN : Nᵀ * N = 1)
    (hβ : β ≤ 1) (hτ0 : 0 ≤ τ)
    (hτ : ∀ X : Matrix q K ℝ,
      frob (R * sqrtDiag d * N * X) (R * sqrtDiag d * N * X) ≤ τ ^ 2 * frob X X) :
    Real.sqrt (frob (R * (-(nullStep d N G β)) + β • (R * Matrix.diagonal d * G))
        (R * (-(nullStep d N G β)) + β • (R * Matrix.diagonal d * G)))
      ≤ (1 - β) * τ * Real.sqrt (frob (sqrtDiag d * G) (sqrtDiag d * G)) := by
  have h := ref_residual_sq R d hd N G β τ hN hτ
  have hc : 0 ≤ (1 - β) * τ := mul_nonneg (sub_nonneg.2 hβ) hτ0
  have h6 := Real.sqrt_le_sqrt h
  rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq hc] at h6
  exact h6

end StableCL
