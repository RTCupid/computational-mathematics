# Lab 2. Numerical Solution of Linear Systems

**English** | [Русский](README.ru.md)

The main report is [numerical_solution_of_linear_systems.ipynb](numerical_solution_of_linear_systems.ipynb).
The executed notebook includes solutions, tables, three figures, and conclusions.
It can be viewed on GitHub and run in VS Code.

**The notebook report, plot labels, and code comments are in Russian.**

## Problem

The lab uses **variant м) (Cyrillic m)** of problem **II.10.5** from
*Practical Classes in Computational Mathematics at MIPT*, Part 1 (2021),
by E. N. Aristova, N. A. Zavyalova, and A. I. Lobanov.
The system is specified on printed page **69**, or page **70 of the PDF**.
The list of required methods follows the instructor's revised assignment.

The matrix is 20 × 20: its main diagonal is 10, its first off-diagonals are 1,
and its second off-diagonals are 0.1; all other entries are zero.
The right-hand side is `b = [1, …, 20]`.
The matrix is symmetric positive definite (SPD) and strictly diagonally dominant.

## Implemented methods

All nine methods are implemented explicitly in [solvers.py](solvers.py):

1. Gaussian elimination with partial pivoting: the largest absolute entry in the current column.
2. LU factorization without pivoting (Doolittle), followed by two triangular solves.
3. Jacobi iteration.
4. Gauss–Seidel iteration.
5. Successive over-relaxation (SOR).
6. Gradient descent with an exact line search, also called steepest descent.
7. The one-dimensional minimal residual iteration.
8. Conjugate gradients (CG).
9. Stabilized biconjugate gradients (BiCGStab).

The minimal residual method here minimizes the residual norm along the current
residual direction; it is not the Lanczos-based MINRES algorithm or GMRES.

`numpy.linalg.solve` is used only for reference solutions and diagnostic
iteration matrices. All methods solve the original system without forming
normal equations or applying preconditioning.

All iterative methods use `x0 = 0`, `rtol = 1e-10`, `atol = 1e-12`, and
`max_iter = 10000`. The stopping criterion is:

```text
||b − A @ x||₂ <= atol + rtol * ||b||₂
```

The history records the true residual recomputed from each approximation,
including the initial guess. CG and BiCGStab may maintain recurrent residual
vectors internally, but stopping decisions and plots use the true residual.

## Running the lab

From the repository root:

```powershell
python -m pip install -r numerical_solution_of_linear_systems/requirements.txt
```

In VS Code, install the Python and Jupyter extensions, open the notebook,
select the Python kernel with the dependencies installed, and click **Run All**.
Save the notebook to preserve the outputs for viewing on GitHub.
A PDF parser, SciPy, and pandas are not required to run the lab.

## Supplying another system

In section 2 of the notebook, replace `A, b = variant_m()` with your own
`np.array(..., dtype=np.float64)` arrays, or load CSV files using `np.loadtxt`.
The same cell sets the initial guess, tolerances, iteration limit, and `OMEGA`.
Rerun the entire notebook after making changes.

The implementation accepts real square systems of any small dimension and uses
dense matrices. CG and steepest descent check for SPD; stationary methods check
the diagonal. The report computes the spectral radii of the Jacobi, Gauss–Seidel,
and SOR iteration matrices.

If a method is inapplicable, breaks down, diverges, or reaches the iteration
limit, its result receives the corresponding status. The last approximation
is not presented as a converged solution. Near-zero checks account for the
scale of the matrix or inner product.

The theoretical justification for variant м) applies to the original matrix.
When replacing it, interpret the results using the new diagnostics.

## Results for variant м)

With the default parameters, all nine methods successfully solve the system.
The condition number is `κ₂(A) ≈ 1.48155`.
The Gaussian elimination and LU residual norms are approximately `4.5e-15`.

| Iterative method | Iterations |
|---|---:|
| Jacobi | 15 |
| Gauss–Seidel | 10 |
| SOR, ω = 1.05 | 12 |
| Steepest descent | 13 |
| Minimal residual | 12 |
| CG | 10 |
| BiCGStab | 6 |

Iteration counts do not directly measure runtime: BiCGStab performs up to two
main matrix–vector products per iteration, plus true-residual checks.
SOR with the chosen parameter is slower than Gauss–Seidel on this system;
a separate experiment explores the effect of ω. Results near machine precision
may vary by platform.

Figures are saved both in the notebook and in [figures](figures):

- `convergence_all.png` — all seven iterative methods on one plot;
- `convergence_individual.png` — a separate panel for each method;
- `sor_parameter.png` — the effect of the relaxation parameter.

All figures use a logarithmic residual-norm axis and a linear iteration axis.
Zero residuals are omitted only from the plots.

## Tests

```powershell
python -m unittest discover -s numerical_solution_of_linear_systems -v
```

[test_solvers.py](test_solvers.py) checks the variant's coefficients, solutions
for the original and another SPD system, a nonsymmetric system, Gaussian pivoting,
the identity `A = L @ U`, singular systems, a zero right-hand side, an exact
initial guess, first-step formulas, stopping statuses, and invalid inputs.
