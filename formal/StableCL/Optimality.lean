-- STATUS: compiled in CI on PR #5 (M4/M5 build + axiom audit).
-- Copyright (c) 2026 Mohamed Alkobrosli
-- SPDX-License-Identifier: MIT
import StableCL.Descent
import StableCL.DegreeControl
import StableCL.RefPreservation

/-!
# M5. The strict step is the unique constrained minimiser   (`math.tex` Prop. 1(c))

**Problem (fixed before any proof).** With `S = {j : 0 < d j}`,

* feasible set `F = {Z : Z_j = 0 for j ∉ S, R Z = 0}`  (`Feasible`),
* objective `J(Z) = ⟨G, Z⟩ + ½ ∑_{j ∈ S} ‖Z_j‖² / d_j`   (`objective`).

The sum is restricted to `S` explicitly (`penalty` uses `if 0 < d j`); it does not rely on Lean's
`x / 0 = 0`. Uniqueness is claimed **over `F`**, not over all of `ℝ^{p×K}`: `J` does not penalise
rows outside `S`, so only the constraint `Z_{S^c} = 0` makes the minimiser unique.

**Hypotheses.** `hd : d ≥ 0` (A0); `hAN : A N = 0` (A2, `A = R D^{1/2}`);
and **`hcomp` (A5, projector completeness): `N Nᵀ V = V` whenever `A V = 0`**. Given `A N = 0` and
orthonormal columns (`NᵀN = I`, which this theorem does not assume), this is equivalent to `N`
spanning all of `null(A)`. A5 is new relative to M1-M3 and is the `tol → 0` hypothesis of `math.tex` Prop. 1 ("`P` is the orthogonal
projector onto `null(A)`"). Without it the claim is false (a smaller basis gives a feasible but
non-minimal step).

**Proof idea.** Put `V = D^{-1/2} Z` on `S`. Then `J(Z) = ⟨u, V⟩ + ½ ‖V‖²` with `u = D^{1/2} G`,
and for feasible `Z, Z*`:  `J(Z) = J(Z*) + ½ ‖V - V*‖²`  (`objective_expand`). The cross term
vanishes because `V - V* ∈ null(A)` is fixed by `P`, and `(I - P) u ⟂ null(A)`.
-/

open Matrix

namespace StableCL

variable {m p q K : Type*} [Fintype p] [Fintype q] [Fintype K] [DecidableEq p] [DecidableEq q]

/-- `D^{-1/2}` on `S = {d_j > 0}`, and `0` off `S`. -/
noncomputable def isqrtDiag (d : p → ℝ) : Matrix p p ℝ :=
  Matrix.diagonal fun j => if 0 < d j then (Real.sqrt (d j))⁻¹ else 0

/-- `½`-penalty `∑_{j ∈ S, k} Z_{jk}² / d_j` (before the factor `½`). -/
noncomputable def penalty (d : p → ℝ) (Z : Matrix p K ℝ) : ℝ :=
  ∑ j, ∑ k, if 0 < d j then Z j k ^ 2 / d j else 0

/-- `J(Z) = ⟨G, Z⟩ + ½ ∑_{j ∈ S} ‖Z_j‖² / d_j`. -/
noncomputable def objective (d : p → ℝ) (G Z : Matrix p K ℝ) : ℝ :=
  frob G Z + penalty d Z / 2

/-- `Z_{S^c} = 0` and `R Z = 0`. -/
structure Feasible (R : Matrix m p ℝ) (d : p → ℝ) (Z : Matrix p K ℝ) : Prop where
  rows : ∀ j, ¬ 0 < d j → ∀ k, Z j k = 0
  ref : R * Z = 0

/-! ### Entrywise facts about the diagonal factors -/

theorem isqrt_mul_apply (d : p → ℝ) (Z : Matrix p K ℝ) (j : p) (k : K) :
    (isqrtDiag d * Z) j k = (if 0 < d j then (Real.sqrt (d j))⁻¹ else 0) * Z j k := by
  simp [isqrtDiag, Matrix.diagonal_mul]

theorem sqrtDiag_mul_apply (d : p → ℝ) (M : Matrix p K ℝ) (j : p) (k : K) :
    (sqrtDiag d * M) j k = Real.sqrt (d j) * M j k := by
  simp [sqrtDiag, Matrix.diagonal_mul]

/-- `D^{1/2} D^{-1/2} Z = Z` when `Z` vanishes off `S`. -/
theorem sqrtDiag_isqrt_mul {d : p → ℝ} {Z : Matrix p K ℝ}
    (hZ : ∀ j, ¬ 0 < d j → ∀ k, Z j k = 0) :
    sqrtDiag d * (isqrtDiag d * Z) = Z := by
  ext j k
  rw [sqrtDiag_mul_apply, isqrt_mul_apply]
  by_cases h : 0 < d j
  · have hs0 : Real.sqrt (d j) ≠ 0 := (Real.sqrt_pos.2 h).ne'
    rw [if_pos h]
    exact mul_inv_cancel_left₀ hs0 _
  · rw [if_neg h, hZ j h k]
    simp

/-- `D^{-1/2} D^{1/2} X = X` on rows of `S`. -/
theorem isqrt_sqrtDiag_apply (d : p → ℝ) {j : p} (h : 0 < d j) (X : Matrix p K ℝ) (k : K) :
    (isqrtDiag d * (sqrtDiag d * X)) j k = X j k := by
  have hs0 : Real.sqrt (d j) ≠ 0 := (Real.sqrt_pos.2 h).ne'
  rw [isqrt_mul_apply, sqrtDiag_mul_apply, if_pos h]
  exact inv_mul_cancel_left₀ hs0 _

/-- The penalty is the squared Frobenius norm of `D^{-1/2} Z`. -/
theorem penalty_eq (d : p → ℝ) (Z : Matrix p K ℝ) :
    penalty d Z = frob (isqrtDiag d * Z) (isqrtDiag d * Z) := by
  rw [frob_apply]
  unfold penalty
  refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => ?_
  rw [isqrt_mul_apply]
  by_cases h : 0 < d j
  · have hs : Real.sqrt (d j) ^ 2 = d j := Real.sq_sqrt h.le
    simp only [if_pos h]
    calc Z j k ^ 2 / d j = Z j k ^ 2 / Real.sqrt (d j) ^ 2 := by rw [hs]
      _ = (Real.sqrt (d j))⁻¹ * Z j k * ((Real.sqrt (d j))⁻¹ * Z j k) := by ring
  · simp only [if_neg h]
    simp

/-- On a feasible `Z`, `J(Z) = ⟨u, V⟩ + ½ ‖V‖²` with `V = D^{-1/2} Z` and `u = D^{1/2} G`. -/
theorem objective_eq (d : p → ℝ) (G : Matrix p K ℝ) {Z : Matrix p K ℝ}
    (hZ : ∀ j, ¬ 0 < d j → ∀ k, Z j k = 0) :
    objective d G Z =
      frob (sqrtDiag d * G) (isqrtDiag d * Z)
        + frob (isqrtDiag d * Z) (isqrtDiag d * Z) / 2 := by
  have h1 : frob G Z = frob (sqrtDiag d * G) (isqrtDiag d * Z) := by
    calc frob G Z = frob G (sqrtDiag d * (isqrtDiag d * Z)) := by rw [sqrtDiag_isqrt_mul hZ]
      _ = frob (sqrtDiag d * G) (isqrtDiag d * Z) := frob_sqrtDiag_right d G _
  unfold objective
  rw [h1, penalty_eq]

/-! ### Algebra of the completed square -/

/-- Two inner products agree if the left factors agree on the support of the right factor. -/
theorem frob_eq_of_agree_on_support {d : p → ℝ} {A A' W : Matrix p K ℝ}
    (hW : ∀ j, ¬ 0 < d j → ∀ k, W j k = 0)
    (hA : ∀ j, 0 < d j → ∀ k, A j k = A' j k) :
    frob A W = frob A' W := by
  rw [frob_apply, frob_apply]
  refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => ?_
  by_cases h : 0 < d j
  · rw [hA j h k]
  · rw [hW j h k]
    simp

/-- `⟨u, V⟩ + ½‖V‖² = (⟨u, V*⟩ + ½‖V*‖²) + ½‖W‖²` when `V = V* + W` and `(u + V*) ⟂ W`. -/
theorem complete_square (u V Vs W : Matrix p K ℝ) (hV : V = Vs + W)
    (horth : frob (u + Vs) W = 0) :
    frob u V + frob V V / 2 = (frob u Vs + frob Vs Vs / 2) + frob W W / 2 := by
  subst hV
  have hA : frob u (Vs + W) = frob u Vs + frob u W := frob_add_right _ _ _
  have hB : frob (Vs + W) (Vs + W) = frob Vs Vs + 2 * frob Vs W + frob W W := by
    rw [frob_add_left, frob_add_right, frob_add_right, frob_comm W Vs]
    ring
  have hC : frob (u + Vs) W = frob u W + frob Vs W := frob_add_left _ _ _
  rw [hA, hB]
  linarith

/-! ### The strict step -/

theorem nullStep_zero_eq (d : p → ℝ) (N : Matrix p q ℝ) (G : Matrix p K ℝ) :
    nullStep d N G 0 = sqrtDiag d * (N * Nᵀ * (sqrtDiag d * G)) := by
  unfold nullStep
  simp

/-- `ΔW = -nullStep d N G 0` is feasible. Uses M2 (frozen rows) and M3 (`A N = 0`). -/
theorem strictStep_feasible (R : Matrix m p ℝ) (d : p → ℝ) (N : Matrix p q ℝ)
    (G : Matrix p K ℝ) (hd : ∀ j, 0 ≤ d j) (hAN : R * sqrtDiag d * N = 0) :
    Feasible R d (-(nullStep d N G 0)) := by
  refine ⟨?_, ?_⟩
  · intro j hj k
    have hj0 : d j = 0 := le_antisymm (not_lt.1 hj) (hd j)
    rw [Matrix.neg_apply, nullStep_row_eq_zero d N G 0 j hj0 k, neg_zero]
  · rw [Matrix.mul_neg, ref_preserved_exact R d N G hAN, neg_zero]

/-- If `Z` is feasible then `D^{-1/2} Z` lies in `null(A)`. -/
theorem isqrt_mem_ker (R : Matrix m p ℝ) (d : p → ℝ) {Z : Matrix p K ℝ}
    (hZ : Feasible R d Z) : R * sqrtDiag d * (isqrtDiag d * Z) = 0 := by
  rw [Matrix.mul_assoc, sqrtDiag_isqrt_mul hZ.rows]
  exact hZ.ref

/-- The key identity: `J(Z) = J(Z*) + ½ ‖D^{-1/2} Z - D^{-1/2} Z*‖²` for feasible `Z`,
where `Z*` is any feasible point whose `D^{-1/2}`-image on `S` is `-P u`. -/
theorem objective_expand (R : Matrix m p ℝ) (d : p → ℝ) (G : Matrix p K ℝ) (N : Matrix p q ℝ)
    (hcomp : ∀ V : Matrix p K ℝ, R * sqrtDiag d * V = 0 → N * Nᵀ * V = V)
    {Z Zs : Matrix p K ℝ} (hZ : Feasible R d Z) (hZs : Feasible R d Zs)
    (hVs : ∀ j, 0 < d j → ∀ k,
      (isqrtDiag d * Zs) j k = -(N * Nᵀ * (sqrtDiag d * G)) j k) :
    objective d G Z = objective d G Zs +
      frob (isqrtDiag d * Z - isqrtDiag d * Zs) (isqrtDiag d * Z - isqrtDiag d * Zs) / 2 := by
  have hAW : R * sqrtDiag d * (isqrtDiag d * Z - isqrtDiag d * Zs) = 0 := by
    rw [Matrix.mul_sub, isqrt_mem_ker R d hZ, isqrt_mem_ker R d hZs, sub_self]
  have hPW := hcomp _ hAW
  have hW0 : ∀ j, ¬ 0 < d j → ∀ k,
      (isqrtDiag d * Z - isqrtDiag d * Zs) j k = 0 := by
    intro j h k
    simp only [Matrix.sub_apply, isqrt_mul_apply, if_neg h, zero_mul, sub_self]
  have hagree : ∀ j, 0 < d j → ∀ k,
      (sqrtDiag d * G + isqrtDiag d * Zs) j k
        = (sqrtDiag d * G - N * Nᵀ * (sqrtDiag d * G)) j k := by
    intro j h k
    rw [Matrix.add_apply, hVs j h k, Matrix.sub_apply, ← sub_eq_add_neg]
  have horth : frob (sqrtDiag d * G + isqrtDiag d * Zs)
      (isqrtDiag d * Z - isqrtDiag d * Zs) = 0 := by
    have key : frob (sqrtDiag d * G + isqrtDiag d * Zs) (isqrtDiag d * Z - isqrtDiag d * Zs)
        = frob (sqrtDiag d * G - N * Nᵀ * (sqrtDiag d * G))
            (isqrtDiag d * Z - isqrtDiag d * Zs) :=
      frob_eq_of_agree_on_support hW0 hagree
    rw [key, frob_sub_left, frob_symm_mul_left (proj_transpose N) (sqrtDiag d * G), hPW,
      sub_self]
  rw [objective_eq d G hZ.rows, objective_eq d G hZs.rows]
  exact complete_square (sqrtDiag d * G) (isqrtDiag d * Z) (isqrtDiag d * Zs)
    (isqrtDiag d * Z - isqrtDiag d * Zs) (by abel) horth

/-- **Prop. 1(c).** `ΔW = -nullStep d N G 0` is feasible, minimises `J` over the feasible set,
and is the only feasible minimiser. -/
theorem strictStep_unique_minimiser (R : Matrix m p ℝ) (d : p → ℝ) (N : Matrix p q ℝ)
    (G : Matrix p K ℝ) (hd : ∀ j, 0 ≤ d j)
    (hAN : R * sqrtDiag d * N = 0)
    (hcomp : ∀ V : Matrix p K ℝ, R * sqrtDiag d * V = 0 → N * Nᵀ * V = V) :
    Feasible R d (-(nullStep d N G 0)) ∧
    (∀ Z, Feasible R d Z →
      objective d G (-(nullStep d N G 0)) ≤ objective d G Z) ∧
    (∀ Z, Feasible R d Z →
      objective d G Z = objective d G (-(nullStep d N G 0)) → Z = -(nullStep d N G 0)) := by
  have hZs := strictStep_feasible R d N G hd hAN
  have hVs : ∀ j, 0 < d j → ∀ k,
      (isqrtDiag d * (-(nullStep d N G 0))) j k = -(N * Nᵀ * (sqrtDiag d * G)) j k := by
    intro j h k
    rw [Matrix.mul_neg, nullStep_zero_eq, Matrix.neg_apply, isqrt_sqrtDiag_apply d h]
  refine ⟨hZs, ?_, ?_⟩
  · intro Z hZ
    have h := objective_expand R d G N hcomp hZ hZs hVs
    have h0 := frob_self_nonneg
      (isqrtDiag d * Z - isqrtDiag d * (-(nullStep d N G 0)))
    linarith
  · intro Z hZ heq
    have h := objective_expand R d G N hcomp hZ hZs hVs
    have hW : frob (isqrtDiag d * Z - isqrtDiag d * (-(nullStep d N G 0)))
        (isqrtDiag d * Z - isqrtDiag d * (-(nullStep d N G 0))) = 0 := by
      linarith
    have hV : isqrtDiag d * Z = isqrtDiag d * (-(nullStep d N G 0)) :=
      sub_eq_zero.1 (frob_self_eq_zero hW)
    calc Z = sqrtDiag d * (isqrtDiag d * Z) := (sqrtDiag_isqrt_mul hZ.rows).symm
      _ = sqrtDiag d * (isqrtDiag d * (-(nullStep d N G 0))) := by rw [hV]
      _ = -(nullStep d N G 0) := sqrtDiag_isqrt_mul hZs.rows

end StableCL
