-- STATUS: DRAFT, not yet compiled.
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

/-- Diagonal rate `d_j = η_{deg j} · m_{deg j}`; degree groups are `Fin 4`. -/
def rate (η mCtl : Fin 4 → ℝ) (deg : p → Fin 4) : p → ℝ :=
  fun j => η (deg j) * mCtl (deg j)

/-- `D^{1/2} = diag(√d_j)`. -/
noncomputable def sqrtDiag [DecidableEq p] (d : p → ℝ) : Matrix p p ℝ :=
  Matrix.diagonal fun j => Real.sqrt (d j)

/-- Ordinary rate-weighted SGD step `S = D G` (the update is `-S`). -/
def sgdStep [Fintype p] [DecidableEq p] (d : p → ℝ) (G : Matrix p K ℝ) : Matrix p K ℝ :=
  Matrix.diagonal d * G

/-- Relaxed null-space step `S = D^{1/2} [β I + (1-β) N Nᵀ] D^{1/2} G` (the update is `-S`).
No property of `N` is assumed in the definition. -/
noncomputable def nullStep [Fintype p] [Fintype q] [DecidableEq p]
    (d : p → ℝ) (N : Matrix p q ℝ) (G : Matrix p K ℝ) (β : ℝ) : Matrix p K ℝ :=
  sqrtDiag d * ((β • (1 : Matrix p p ℝ) + (1 - β) • (N * Nᵀ)) * (sqrtDiag d * G))

end StableCL
