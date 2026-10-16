"""Численное дифференцирование: пять функций и пять разностных формул."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FUNCTIONS = (
    ("sin_x2", r"\sin(x^2)", lambda x: np.sin(x**2),
     lambda x: 2 * x * np.cos(x**2)),
    ("cos_sin_x", r"\cos(\sin x)", lambda x: np.cos(np.sin(x)),
     lambda x: -np.sin(np.sin(x)) * np.cos(x)),
    ("exp_sin_cos_x", r"\exp(\sin(\cos x))", lambda x: np.exp(np.sin(np.cos(x))),
     lambda x: -np.exp(np.sin(np.cos(x))) * np.cos(np.cos(x)) * np.sin(x)),
    ("log_x_plus_3", r"\ln(x+3)", lambda x: np.log(x + 3),
     lambda x: 1 / (x + 3)),
    ("sqrt_x_plus_3", r"\sqrt{x+3}", lambda x: np.sqrt(x + 3),
     lambda x: 1 / (2 * np.sqrt(x + 3))),
)

METHOD_LABELS = (
    "1. Правая разность, O(h)",
    "2. Левая разность, O(h)",
    "3. Центральная разность, O(h²)",
    "4. Формула четвёртого порядка, O(h⁴)",
    "5. Формула шестого порядка, O(h⁶)",
)


def approximate_derivatives(f, x, h):
    """Вернуть массив (5, len(h)), сохраняя вид формул из задания."""
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 1 or h.size == 0 or not np.all(np.isfinite(h) & (h > 0)):
        raise ValueError("Шаги должны быть непустым одномерным массивом положительных чисел.")
    d1 = (f(x + h) - f(x - h)) / (2 * h)
    d2 = (f(x + 2 * h) - f(x - 2 * h)) / (4 * h)
    d3 = (f(x + 3 * h) - f(x - 3 * h)) / (6 * h)
    return np.stack((
        (f(x + h) - f(x)) / h,
        (f(x) - f(x - h)) / h,
        d1,
        4 / 3 * d1 - 1 / 3 * d2,
        3 / 2 * d1 - 3 / 5 * d2 + 1 / 10 * d3,
    ))


def calculate(x=1.0):
    """Вычислить абсолютные погрешности для n=1,...,21 в float64."""
    h = 2.0 / 2.0 ** np.arange(1, 22)
    # Самый широкий шаблон обращается к x-3h; ln требует x-3h > -3.
    if not np.isfinite(x) or x - 3 * h.max() <= -3:
        raise ValueError("Для всех пяти функций и всех шагов требуется конечная точка x > 0.")
    results = []
    for slug, formula, f, derivative in FUNCTIONS:
        exact = float(derivative(x))
        approximations = approximate_derivatives(f, x, h)
        results.append({
            "slug": slug, "formula": formula, "exact": exact,
            "approximations": approximations,
            "errors": np.abs(approximations - exact),
        })
    return h, results


def plot_errors(h, result, x=1.0, output_dir=None):
    """Одна фигура с пятью линиями; нули не подменяются положительными числами."""
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    markers = ("o", "s", "^", "D", "v")
    for label, marker, error in zip(METHOD_LABELS, markers, result["errors"]):
        # log(0) не определён: пропускаем такие точки только при отображении.
        visible = np.where(error > 0, error, np.nan)
        ax.loglog(h, visible, marker=marker, markersize=4, linewidth=1.4, label=label)
    ax.set(title=rf"$f(x)={result['formula']}$, $x_0={x:g}$",
           xlabel="Шаг численного дифференцирования h",
           ylabel="Абсолютная погрешность |Dₕf(x₀) − f′(x₀)|")
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)
    ax.legend(fontsize=9)
    zeros = np.count_nonzero(result["errors"] == 0)
    if zeros:
        fig.suptitle(f"Нулевые погрешности ({zeros}) пропущены на логарифмической оси.",
                     fontsize=9)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / f"{result['slug']}.png", dpi=160)
    return fig


if __name__ == "__main__":
    steps, computed = calculate()
    for item in computed:
        plot_errors(steps, item, output_dir=Path(__file__).parent / "figures")
    plt.show()
