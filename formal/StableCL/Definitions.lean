-- STATUS: compiled in CI on PR #3, including `rate_nonneg` and `sqrtDiag_mul_self`.
-- Copyright (c) 2026 Mohamed Alkobrosli
-- SPDX-License-Identifier: MIT
import Mathlib

/-!
# Definitions mirrored from `math.tex` and `stable_cl/optimization.py`

Everything is over `ℝ`. Index types are arbitrary finite types, so nothing here depends on how
`stable_cl.features.poly_features` enumerates monomials.

* `rate`      : `d_j = η_{deg j} · m_{deg j}`            (`diagonal_rates`)
* `sqrtDiag`  : `D^{1/2} = diag(√d_j)`
* `sgdStep`   : `D G`                                     (unsigned; `ΔW = -D G`)
* `nullStep`  : `D^{1/2} [β I + (1-β) N Nᵀ] D^{1/2} G`    (`null_step`, unsigned; `ΔW = -S`)

`math.tex` writes the middle factor as `P + β(I - P)` with `P = N Nᵀ`; this equals
`β I + (1-β) P`, the form used in `null_step`, so the two descriptions agree.
-/

open Matrix

namespace StableCL

variable {p q K : Type*}

/-- Diagonal rate `d_j = η_{deg j} · m_{deg j}`; degree groups are `Fin 4`.

Domain: `math.tex` and `diagonal_rates` require `η_g ≥ 0` and `0 ≤ m_g ≤ 1`, hence `d_j ≥ 0`
(see `rate_nonneg`). The definition itself accepts any real inputs. -/
def rate (η mCtl : Fin 4 → ℝ) (deg : p → Fin 4) : p → ℝ :=
  fun j => η (deg j) * mCtl (deg j)

/-- `η_g ≥ 0` and `m_g ≥ 0` give `d_j ≥ 0`. The upper bound `m_g ≤ 1` is not needed for
nonnegativity. -/
theorem rate_nonneg (η mCtl : Fin 4 → ℝ) (deg : p → Fin 4)
    (hη : ∀ g, 0 ≤ η g) (hm : ∀ g, 0 ≤ mCtl g) (j : p) :
    0 ≤ rate η mCtl deg j := by
  unfold rate
  exact mul_nonneg (hη _) (hm _)

/-- `D^{1/2} = diag(√d_j)`.

`Real.sqrt` sends negative numbers to `0`, so for a negative `d_j` this definition silently
differs from the Python code, which rejects negative rates. The M1-M3 theorems hold for every
real `d` and do not need `d ≥ 0`. Results that use `D = D^{1/2} D^{1/2}` (planned M4/M5) must
take `∀ j, 0 ≤ d j` as an explicit hypothesis; see `sqrtDiag_mul_self`. -/
noncomputable def sqrtDiag [DecidableEq p] (d : p → ℝ) : Matrix p p ℝ :=
  Matrix.diagonal fun j => Real.sqrt (d j)

/-- Under `d ≥ 0`, `D^{1/2}` really is a square root of `D`. -/
theorem sqrtDiag_mul_self [Fintype p] [DecidableEq p] (d : p → ℝ) (hd : ∀ j, 0 ≤ d j) :
    sqrtDiag d * sqrtDiag d = Matrix.diagonal d := by
  unfold sqrtDiag
  rw [Matrix.diagonal_mul_diagonal]
  exact congrArg Matrix.diagonal (funext fun j => Real.mul_self_sqrt (hd j))

/-- Ordinary rate-weighted SGD step `S = D G` (the update is `-S`). -/
def sgdStep [Fintype p] [DecidableEq p] (d : p → ℝ) (G : Matrix p K ℝ) : Matrix p K ℝ :=
  Matrix.diagonal d * G

/-- Relaxed null-space step `S = D^{1/2} [β I + (1-β) N Nᵀ] D^{1/2} G` (the update is `-S`).
No property of `N` is assumed in the definition. -/
noncomputable def nullStep [Fintype p] [Fintype q] [DecidableEq p]
    (d : p → ℝ) (N : Matrix p q ℝ) (G : Matrix p K ℝ) (β : ℝ) : Matrix p K ℝ :=
  sqrtDiag d * ((β • (1 : Matrix p p ℝ) + (1 - β) • (N * Nᵀ)) * (sqrtDiag d * G))

end StableCL
