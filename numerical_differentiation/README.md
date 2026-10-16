# Lab 1. Numerical Differentiation

**English** | [Русский](README.ru.md)

The main report is [numerical_differentiation.ipynb](numerical_differentiation.ipynb).
It contains the assignment, analytical derivatives, five finite-difference
formulas, five executed plots, and conclusions. Each plot compares five methods
with logarithmic scales on both axes. PNG images are also available in [figures](figures).

**The notebook report, plot labels, and code comments are in Russian.**

## Problem and methods

The lab approximates the first derivatives of five functions:

- `sin(x²)`;
- `cos(sin(x))`;
- `exp(sin(cos(x)))`;
- `ln(x + 3)`;
- `sqrt(x + 3)`.

It compares forward and backward differences, the central difference,
and fourth- and sixth-order central formulas. Their truncation error orders
are **1, 1, 2, 4, and 6**, respectively.

The evaluation point is **x₀ = 1**, since the assignment image does not specify
one. Change `X0` in the notebook to use another point. For all functions and
the full set of steps, `X0 > 0` is required to keep every logarithm argument
positive. The steps follow the assignment: `h = 2 / 2**n`, `n = 1, …, 21`.

## Running in VS Code

1. Install Microsoft's **Python** and **Jupyter** extensions.
2. Install the dependencies from the repository root:

   ```powershell
   python -m pip install -r numerical_differentiation/requirements.txt
   ```

3. Open the notebook, choose **Select Kernel**, select the Python environment
   with the installed dependencies, and click **Run All**. Save the notebook
   to retain its outputs.

GitHub displays saved notebook cells and plots without running Python.
Recomputing the results requires a local execution environment.

## Running without a notebook

```powershell
python numerical_differentiation/differentiation.py
```

The script saves five PNG files and opens five separate figures when a graphical
Matplotlib backend is available. In the notebook, each figure appears in its own cell.

## Numerical accuracy and validation

Calculations use `float64`. Absolute errors are measured against analytical
derivatives evaluated in the same arithmetic. A computed zero error remains
zero in the data but is omitted from the logarithmic plot; no artificial
positive floor is introduced.

The plots illustrate both truncation error and the effect of roundoff at small
steps. A smaller step does not always produce a more accurate derivative.

The notebook has been executed with a Jupyter kernel: all seven code cells
completed successfully, with five figures and text outputs saved in the report.
