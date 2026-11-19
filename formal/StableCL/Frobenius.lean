-- STATUS: DRAFT, not yet compiled.
-- Copyright (c) 2026 Mohamed Alkobrosli
-- SPDX-License-Identifier: MIT
import StableCL.Definitions

/-!
# Frobenius inner product (shared by M4 and M5)

`frob A B = tr(Aᵀ B) = ∑ j, ∑ k, A j k * B j k`, the inner product `⟨A, B⟩` of `math.tex`.
`‖A‖_F² = frob A A`. It is defined directly (not through Mathlib's matrix norms) so that every
fact used later is an elementary identity about sums or traces.
-/

open Matrix

namespace StableCL

variable {p K : Type*} [Fintype p] [Fintype K]

/-- `⟨A, B⟩ = tr(Aᵀ B)`. -/
def frob (A B : Matrix p K ℝ) : ℝ := Matrix.trace (Aᵀ * B)

theorem frob_apply (A B : Matrix p K ℝ) : frob A B = ∑ j, ∑ k, A j k * B j k := by
  unfold frob
  simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply]
  exact Finset.sum_comm

theorem frob_comm (A B : Matrix p K ℝ) : frob A B = frob B A := by
  rw [frob_apply, frob_apply]
  refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => ?_
  exact mul_comm _ _

theorem frob_zero_right (A : Matrix p K ℝ) : frob A 0 = 0 := by
  unfold frob
  rw [Matrix.mul_zero, Matrix.trace_zero]

theorem frob_add_right (A B C : Matrix p K ℝ) : frob A (B + C) = frob A B + frob A C := by
  unfold frob
  rw [Matrix.mul_add, Matrix.trace_add]

theorem frob_sub_right (A B C : Matrix p K ℝ) : frob A (B - C) = frob A B - frob A C := by
  unfold frob
  rw [Matrix.mul_sub, Matrix.trace_sub]

theorem frob_neg_right (A B : Matrix p K ℝ) : frob A (-B) = -frob A B := by
  unfold frob
  rw [Matrix.mul_neg, Matrix.trace_neg]

theorem frob_smul_right (c : ℝ) (A B : Matrix p K ℝ) : frob A (c • B) = c * frob A B := by
  unfold frob
  rw [Matrix.mul_smul, Matrix.trace_smul, smul_eq_mul]

theorem frob_add_left (A B C : Matrix p K ℝ) : frob (A + B) C = frob A C + frob B C := by
  rw [frob_comm (A + B) C, frob_add_right, frob_comm C A, frob_comm C B]

theorem frob_sub_left (A B C : Matrix p K ℝ) : frob (A - B) C = frob A C - frob B C := by
  rw [frob_comm (A - B) C, frob_sub_right, frob_comm C A, frob_comm C B]

theorem frob_neg_left (A B : Matrix p K ℝ) : frob (-A) B = -frob A B := by
  rw [frob_comm (-A) B, frob_neg_right, frob_comm B A]

theorem frob_smul_left (c : ℝ) (A B : Matrix p K ℝ) : frob (c • A) B = c * frob A B := by
  rw [frob_comm (c • A) B, frob_smul_right, frob_comm B A]

theorem frob_neg_neg (A B : Matrix p K ℝ) : frob (-A) (-B) = frob A B := by
  rw [frob_neg_left, frob_neg_right, neg_neg]

theorem frob_smul_smul (c : ℝ) (A B : Matrix p K ℝ) :
    frob (c • A) (c • B) = c ^ 2 * frob A B := by
  rw [frob_smul_left, frob_smul_right]
  ring

theorem frob_self_nonneg (A : Matrix p K ℝ) : 0 ≤ frob A A := by
  rw [frob_apply]
  exact Finset.sum_nonneg fun j _ => Finset.sum_nonneg fun k _ => mul_self_nonneg _

theorem frob_self_eq_zero {A : Matrix p K ℝ} (h : frob A A = 0) : A = 0 := by
  rw [frob_apply] at h
  have h1 := (Finset.sum_eq_zero_iff_of_nonneg
    (fun j _ => Finset.sum_nonneg fun k _ => mul_self_nonneg (A j k))).1 h
  ext j k
  have h2 := (Finset.sum_eq_zero_iff_of_nonneg
    (fun k _ => mul_self_nonneg (A j k))).1 (h1 j (Finset.mem_univ j))
  have h3 := mul_self_eq_zero.1 (h2 k (Finset.mem_univ k))
  simpa using h3

/-- A symmetric left factor can be moved across the inner product. -/
theorem frob_symm_mul_left {S : Matrix p p ℝ} (hS : Sᵀ = S) (X Y : Matrix p K ℝ) :
    frob (S * X) Y = frob X (S * Y) := by
  unfold frob
  simp only [Matrix.transpose_mul, hS, Matrix.mul_assoc]

section Diagonal
variable [DecidableEq p]

theorem sqrtDiag_transpose (d : p → ℝ) : (sqrtDiag d)ᵀ = sqrtDiag d := by
  unfold sqrtDiag
  exact Matrix.diagonal_transpose _

theorem frob_sqrtDiag_right (d : p → ℝ) (G X : Matrix p K ℝ) :
    frob G (sqrtDiag d * X) = frob (sqrtDiag d * G) X :=
  (frob_symm_mul_left (sqrtDiag_transpose d) G X).symm

end Diagonal

end StableCL
