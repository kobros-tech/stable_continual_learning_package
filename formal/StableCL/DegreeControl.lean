-- STATUS: compiled in CI on PR #3 (M0-M3 build + axiom audit); unchanged since.
-- Copyright (c) 2026 Mohamed Alkobrosli
-- SPDX-License-Identifier: MIT
import StableCL.Definitions

/-!
# M2. Zero effective rate freezes a row   (`math.tex` Prop. 1(b), §2)

Part 1 (frozen rows) is about an arbitrary rate vector `d`.
Part 2 (prefix stability) is a *separate* property: it holds only under the explicit hypothesis
`hext` that the degree map of the expanded model extends the old one. That hypothesis is a fact
about the feature ordering, assumption A1 in `docs/traceability.md`, checked by pytest.
-/

open Matrix

namespace StableCL

variable {p q K : Type*}

/-! ### Part 1: frozen rows -/

section Frozen
variable [Fintype p] [DecidableEq p]

theorem sgdStep_row_eq_zero (d : p → ℝ) (G : Matrix p K ℝ) (j : p) (hj : d j = 0) (k : K) :
    sgdStep d G j k = 0 := by
  simp [sgdStep, hj]

theorem nullStep_row_eq_zero [Fintype q] (d : p → ℝ) (N : Matrix p q ℝ) (G : Matrix p K ℝ)
    (β : ℝ) (j : p) (hj : d j = 0) (k : K) :
    nullStep d N G β j k = 0 := by
  simp [nullStep, sqrtDiag, hj]

end Frozen

/-- `m_g = 0` for the degree of feature `j` gives `d_j = 0`. -/
theorem rate_eq_zero_of_control_zero (η mCtl : Fin 4 → ℝ) (deg : p → Fin 4) (j : p)
    (h : mCtl (deg j) = 0) : rate η mCtl deg j = 0 := by
  simp [rate, h]

/-- Degree control `m_g = 0` freezes every feature of degree `g` under the null-space step. -/
theorem nullStep_frozen_of_control_zero [Fintype p] [Fintype q] [DecidableEq p]
    (η mCtl : Fin 4 → ℝ) (deg : p → Fin 4) (N : Matrix p q ℝ) (G : Matrix p K ℝ) (β : ℝ)
    (j : p) (h : mCtl (deg j) = 0) (k : K) :
    nullStep (rate η mCtl deg) N G β j k = 0 :=
  nullStep_row_eq_zero _ N G β j (rate_eq_zero_of_control_zero η mCtl deg j h) k

/-! ### Part 2: prefix stability (separate; explicit hypothesis) -/

/-- If expansion leaves the degree of every old feature unchanged (`hext`), old features keep
their rate. -/
theorem rate_prefix_stable (η mCtl : Fin 4 → ℝ) (deg : p → Fin 4) (deg' : p ⊕ q → Fin 4)
    (hext : ∀ j, deg' (Sum.inl j) = deg j) (j : p) :
    rate η mCtl deg' (Sum.inl j) = rate η mCtl deg j := by
  simp [rate, hext]

/-- A feature added by expansion takes the rate of its own degree. -/
theorem rate_new_feature (η mCtl : Fin 4 → ℝ) (deg' : p ⊕ q → Fin 4) (i : q) :
    rate η mCtl deg' (Sum.inr i) = η (deg' (Sum.inr i)) * mCtl (deg' (Sum.inr i)) :=
  rfl

end StableCL
