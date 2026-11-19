-- STATUS: DRAFT, not yet compiled.
-- Copyright (c) 2026 Mohamed Alkobrosli
-- SPDX-License-Identifier: MIT
import StableCL.Frobenius

/-!
# M4a. Descent identity and the `β = 1` case   (`math.tex` Prop. 1(d), Prop. 2(a), 2(b))

Hypotheses are explicit and minimal:

* `hN : Nᵀ N = 1` (A3) is needed for the identities that use `P = N Nᵀ` as an orthogonal
  projector (`P² = P = Pᵀ`).
* `hd : ∀ j, 0 ≤ d j` (A0) is needed only to identify `D^{1/2} D^{1/2}` with `D` (the `β = 1` case).

Nothing here assumes `A N = 0`: descent holds for any orthonormal `N`.

With `u = D^{1/2} G`, the unsigned step is `S = nullStep d N G β` and the update is `ΔW = -S`.
Identity proved: `⟨G, S⟩ = β ‖u‖² + (1-β) ‖P u‖²`, so `⟨G, ΔW⟩ = -(β ‖u‖² + (1-β) ‖P u‖²)`.
The conclusion is about the directional derivative of the linearised loss only; it says nothing
about the decrease of the nonlinear loss under a finite step.
-/

open Matrix

namespace StableCL

variable {p q K : Type*} [Fintype p] [Fintype q] [Fintype K] [DecidableEq p] [DecidableEq q]

theorem proj_transpose (N : Matrix p q ℝ) : (N * Nᵀ)ᵀ = N * Nᵀ := by
  rw [Matrix.transpose_mul, Matrix.transpose_transpose]

theorem proj_idem (N : Matrix p q ℝ) (hN : Nᵀ * N = 1) : N * Nᵀ * (N * Nᵀ) = N * Nᵀ := by
  calc N * Nᵀ * (N * Nᵀ) = N * (Nᵀ * N) * Nᵀ := by simp only [Matrix.mul_assoc]
    _ = N * Nᵀ := by rw [hN, Matrix.mul_one]

/-- `‖P u‖² = ⟨u, P u⟩` for a symmetric idempotent `P`. -/
theorem frob_proj {P : Matrix p p ℝ} (hPT : Pᵀ = P) (hPP : P * P = P) (u : Matrix p K ℝ) :
    frob (P * u) (P * u) = frob u (P * u) := by
  rw [frob_symm_mul_left hPT u (P * u), ← Matrix.mul_assoc, hPP]

/-- `‖P u‖² ≤ ‖u‖²` (Pythagoras for a symmetric idempotent `P`). -/
theorem frob_proj_le {P : Matrix p p ℝ} (hPT : Pᵀ = P) (hPP : P * P = P) (u : Matrix p K ℝ) :
    frob u (P * u) ≤ frob u u := by
  have h : frob (u - P * u) (u - P * u) = frob u u - frob u (P * u) := by
    rw [frob_sub_left, frob_sub_right, frob_sub_right, frob_comm (P * u) u,
      frob_proj hPT hPP u]
    ring
  have := frob_self_nonneg (u - P * u)
  linarith

/-- `‖Nᵀ u‖² ≤ ‖u‖²` for orthonormal columns. -/
theorem frob_transpose_le (N : Matrix p q ℝ) (hN : Nᵀ * N = 1) (u : Matrix p K ℝ) :
    frob (Nᵀ * u) (Nᵀ * u) ≤ frob u u := by
  have h1 : frob (Nᵀ * u) (Nᵀ * u) = frob u (N * Nᵀ * u) := by
    unfold frob
    simp only [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
  rw [h1]
  exact frob_proj_le (proj_transpose N) (proj_idem N hN) u

/-- The descent identity (Prop. 2(b)), for the unsigned step. -/
theorem nullStep_inner (d : p → ℝ) (N : Matrix p q ℝ) (G : Matrix p K ℝ) (β : ℝ)
    (hN : Nᵀ * N = 1) :
    frob G (nullStep d N G β) =
      β * frob (sqrtDiag d * G) (sqrtDiag d * G)
        + (1 - β) * frob (N * Nᵀ * (sqrtDiag d * G)) (N * Nᵀ * (sqrtDiag d * G)) := by
  have hM : (β • (1 : Matrix p p ℝ) + (1 - β) • (N * Nᵀ)) * (sqrtDiag d * G)
      = β • (sqrtDiag d * G) + (1 - β) • (N * Nᵀ * (sqrtDiag d * G)) := by
    simp only [Matrix.add_mul, Matrix.smul_mul, Matrix.one_mul]
  have h1 : frob G (nullStep d N G β) =
      frob (sqrtDiag d * G)
        ((β • (1 : Matrix p p ℝ) + (1 - β) • (N * Nᵀ)) * (sqrtDiag d * G)) := by
    unfold nullStep
    exact frob_sqrtDiag_right d G _
  rw [h1, hM, frob_add_right, frob_smul_right, frob_smul_right,
    frob_proj (proj_transpose N) (proj_idem N hN)]

/-- Signed form: `⟨G, ΔW⟩ = -(β ‖u‖² + (1-β) ‖P u‖²)` with `ΔW = -nullStep`. -/
theorem deltaW_inner (d : p → ℝ) (N : Matrix p q ℝ) (G : Matrix p K ℝ) (β : ℝ)
    (hN : Nᵀ * N = 1) :
    frob G (-(nullStep d N G β)) =
      -(β * frob (sqrtDiag d * G) (sqrtDiag d * G)
        + (1 - β) * frob (N * Nᵀ * (sqrtDiag d * G)) (N * Nᵀ * (sqrtDiag d * G))) := by
  rw [frob_neg_right, nullStep_inner d N G β hN]

/-- Non-ascent: `⟨G, ΔW⟩ ≤ 0` for `0 ≤ β ≤ 1`. -/
theorem deltaW_inner_nonpos (d : p → ℝ) (N : Matrix p q ℝ) (G : Matrix p K ℝ) (β : ℝ)
    (hN : Nᵀ * N = 1) (hβ0 : 0 ≤ β) (hβ1 : β ≤ 1) :
    frob G (-(nullStep d N G β)) ≤ 0 := by
  rw [deltaW_inner d N G β hN]
  have h1 : 0 ≤ 1 - β := sub_nonneg.2 hβ1
  have h2 := mul_nonneg hβ0 (frob_self_nonneg (sqrtDiag d * G))
  have h3 := mul_nonneg h1 (frob_self_nonneg (N * Nᵀ * (sqrtDiag d * G)))
  linarith

/-- Strict descent when `β > 0` and `u ≠ 0`. -/
theorem deltaW_inner_neg (d : p → ℝ) (N : Matrix p q ℝ) (G : Matrix p K ℝ) (β : ℝ)
    (hN : Nᵀ * N = 1) (hβ0 : 0 < β) (hβ1 : β ≤ 1) (hu : sqrtDiag d * G ≠ 0) :
    frob G (-(nullStep d N G β)) < 0 := by
  rw [deltaW_inner d N G β hN]
  have h1 : 0 ≤ 1 - β := sub_nonneg.2 hβ1
  have hpos : 0 < frob (sqrtDiag d * G) (sqrtDiag d * G) :=
    lt_of_le_of_ne (frob_self_nonneg _) (fun h => hu (frob_self_eq_zero h.symm))
  have h2 := mul_pos hβ0 hpos
  have h3 := mul_nonneg h1 (frob_self_nonneg (N * Nᵀ * (sqrtDiag d * G)))
  linarith

/-- The strict step stalls iff `P u = 0` (Prop. 1(d)). -/
theorem nullStep_zero_eq_zero_iff (d : p → ℝ) (N : Matrix p q ℝ) (G : Matrix p K ℝ)
    (hN : Nᵀ * N = 1) :
    nullStep d N G 0 = 0 ↔ N * Nᵀ * (sqrtDiag d * G) = 0 := by
  constructor
  · intro h
    have key := nullStep_inner d N G 0 hN
    rw [h, frob_zero_right] at key
    apply frob_self_eq_zero
    linarith
  · intro h
    have e : (0 : ℝ) • (1 : Matrix p p ℝ) + (1 - 0) • (N * Nᵀ) = N * Nᵀ := by simp
    unfold nullStep
    rw [e, h, Matrix.mul_zero]

/-- `β = 1` is ordinary rate-weighted SGD, for every `N` (Prop. 2(a)). Needs `d ≥ 0` (A0). -/
theorem nullStep_one (d : p → ℝ) (N : Matrix p q ℝ) (G : Matrix p K ℝ)
    (hd : ∀ j, 0 ≤ d j) :
    nullStep d N G 1 = sgdStep d G := by
  unfold nullStep sgdStep
  rw [sub_self, zero_smul, add_zero, one_smul, Matrix.one_mul, ← Matrix.mul_assoc,
    sqrtDiag_mul_self d hd]

end StableCL
