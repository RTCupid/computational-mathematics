"""Учебные реализации прямых и итерационных методов решения Ax=b.

Все вычисления — float64. Готовые решатели NumPy используются только
в diagnostics(), а не в реализациях девяти методов.
"""

from dataclasses import dataclass

import numpy as np


METHOD_NAMES = {
    "jacobi": "Якоби",
    "gauss_seidel": "Зейдель",
    "sor": "Верхняя релаксация (SOR)",
    "steepest_descent": "Наискорейший спуск",
    "minimal_residual": "Минимальные невязки",
    "cg": "Сопряжённые градиенты (CG)",
    "bicgstab": "BiCGStab",
}


def variant_m():
    """II.10.5, м): n=20, c=10, b=d=1, a=e=0.1, f_i=i (i=1..20)."""
    n = 20
    A = (10 * np.eye(n) + np.diag(np.ones(n - 1), 1)
         + np.diag(np.ones(n - 1), -1)
         + np.diag(np.full(n - 2, 0.1), 2)
         + np.diag(np.full(n - 2, 0.1), -2))
    return A, np.arange(1, n + 1, dtype=np.float64)


def validate_system(A, b):
    """Проверить размеры/конечность и скопировать вещественную СЛАУ."""
    if np.iscomplexobj(A) or np.iscomplexobj(b):
        raise ValueError("Реализация предназначена для вещественных СЛАУ.")
    A = np.array(A, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
        raise ValueError("A должна быть непустой квадратной матрицей.")
    if b.shape != (len(A),):
        raise ValueError("b должен быть одномерным вектором длины len(A).")
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("A и b должны содержать только конечные числа.")
    return A, b


def pivot_tolerance(A):
    return np.finfo(np.float64).eps * len(A) * np.linalg.norm(A, np.inf)


def is_spd(A):
    """Численная проверка симметрии и положительной определённости."""
    tol = pivot_tolerance(A)
    return (np.max(np.abs(A - A.T)) <= tol
            and np.linalg.eigvalsh(A).min() > tol)


def triangular_solve(T, b, *, lower=False):
    """Прямая либо обратная подстановка; входные данные уже проверены."""
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    tol = pivot_tolerance(T)
    for i in (range(n) if lower else range(n - 1, -1, -1)):
        if abs(T[i, i]) <= tol:
            raise np.linalg.LinAlgError("Нулевой или численно малый ведущий элемент.")
        part = T[i, :i] @ x[:i] if lower else T[i, i + 1:] @ x[i + 1:]
        x[i] = (b[i] - part) / T[i, i]
    return x


def gaussian_elimination(A, b):
    """Гаусс с частичным выбором главного элемента: максимум в столбце."""
    A, b = validate_system(A, b)
    n, tol = len(b), pivot_tolerance(A)
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(A[k:, k])))
        if abs(A[pivot, k]) <= tol:
            raise np.linalg.LinAlgError("Матрица вырождена или численно близка к вырожденной.")
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k + 1:] -= factor * A[k, k + 1:]
            b[i] -= factor * b[k]
            A[i, k] = 0.0
    return triangular_solve(A, b)


def lu_factor(A):
    """A=LU без перестановок, схема Дулиттла, diag(L)=1.

    Если ведущий элемент мал, разложение не применяется; система не меняется.
    Для SPD-матрицы все ведущие главные миноры положительны.
    """
    A, _ = validate_system(A, np.zeros(len(A)))
    n, tol = len(A), pivot_tolerance(A)
    L, U = np.eye(n), np.zeros_like(A)
    for k in range(n):
        U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
        if abs(U[k, k]) <= tol:
            raise np.linalg.LinAlgError("LU без перестановок неприменимо: малый ведущий элемент.")
        L[k + 1:, k] = (A[k + 1:, k] - L[k + 1:, :k] @ U[:k, k]) / U[k, k]
    return L, U


def lu_solve(A, b):
    A, b = validate_system(A, b)
    L, U = lu_factor(A)
    return triangular_solve(U, triangular_solve(L, b, lower=True))


@dataclass
class IterationResult:
    x: np.ndarray
    residuals: np.ndarray
    status: str
    message: str

    @property
    def iterations(self):
        return len(self.residuals) - 1

    @property
    def converged(self):
        return self.status == "converged"


def iterate(A, b, method, *, x0=None, rtol=1e-10, atol=1e-12,
            max_iter=10000, omega=1.05):
    """Семь методов с общим критерием ||b-Ax||₂ <= atol+rtol*||b||₂.

    residuals[0] соответствует x0; далее записана истинная невязка каждого
    принятого приближения. В CG/BiCGStab рекуррентная невязка используется
    для направления, но никогда не подменяет невязку для критерия останова.
    Неприменимость, breakdown и лимит итераций отражаются в status.
    """
    A, b = validate_system(A, b)
    if method not in METHOD_NAMES:
        raise ValueError(f"Неизвестный метод: {method}")
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0 or rtol + atol == 0:
        raise ValueError("Допуски должны быть конечными, неотрицательными и не оба нулевыми.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter должен быть неотрицательным целым числом.")
    if x0 is not None and np.iscomplexobj(x0):
        raise ValueError("x0 должен быть вещественным.")
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    if x.shape != b.shape or not np.isfinite(x).all():
        raise ValueError("x0 должен быть конечным вектором той же формы, что b.")
    if method == "sor" and (not np.isfinite(omega) or not 0 < omega < 2):
        raise ValueError("Для SOR требуется 0 < omega < 2; верхняя релаксация: 1 < omega < 2.")
    target = atol + rtol * np.linalg.norm(b)
    r = b - A @ x
    history = [float(np.linalg.norm(r))]

    def finish(status, message):
        return IterationResult(x.copy(), np.array(history), status, message)

    if history[-1] <= target:
        return finish("converged", "Начальное приближение удовлетворяет допуску.")
    if method in {"steepest_descent", "cg"} and not is_spd(A):
        return finish("not_applicable", "Требуется симметричная положительно определённая A.")
    if method in {"jacobi", "gauss_seidel", "sor"} and np.any(np.abs(np.diag(A)) <= pivot_tolerance(A)):
        return finish("not_applicable", "На диагонали есть нулевой или численно малый элемент.")

    # Состояние CG и BiCGStab. Предобусловливание не используется.
    p, shadow = r.copy(), r.copy()
    v = np.zeros_like(r)
    rho_old = alpha = stabilization = 1.0
    eps = np.finfo(np.float64).eps

    def nearly_orthogonal(a, c):
        return abs(a @ c) <= eps * np.linalg.norm(a) * np.linalg.norm(c)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in range(1, max_iter + 1):
            if method == "jacobi":
                new_x = x + r / np.diag(A)
            elif method in {"gauss_seidel", "sor"}:
                w = omega if method == "sor" else 1.0
                new_x = x.copy()
                for i in range(len(b)):
                    sigma = A[i, :i] @ new_x[:i] + A[i, i + 1:] @ x[i + 1:]
                    new_x[i] = (1 - w) * x[i] + w * (b[i] - sigma) / A[i, i]
            elif method in {"steepest_descent", "minimal_residual"}:
                Ar = A @ r
                if nearly_orthogonal(r, Ar):
                    return finish("breakdown", "Скалярное произведение (r, Ar) численно равно нулю.")
                if method == "steepest_descent":
                    step = (r @ r) / (r @ Ar)
                else:
                    step = (r @ Ar) / (Ar @ Ar)
                new_x = x + step * r
            elif method == "cg":
                Ap = A @ p
                if p @ Ap <= 0 or nearly_orthogonal(p, Ap):
                    return finish("breakdown", "Неположительный или численно малый (p, Ap).")
                rr = r @ r
                step = rr / (p @ Ap)
                new_x = x + step * p
                next_r = r - step * Ap
                p = next_r + (next_r @ next_r) / rr * p
                r = next_r
            else:  # BiCGStab
                rho = shadow @ r
                if nearly_orthogonal(shadow, r):
                    return finish("breakdown", "BiCGStab: (r_shadow, r) численно равно нулю.")
                if k == 1:
                    p = r.copy()
                else:
                    if stabilization == 0:
                        return finish("breakdown", "BiCGStab: параметр стабилизации равен нулю.")
                    beta = (rho / rho_old) * (alpha / stabilization)
                    p = r + beta * (p - stabilization * v)
                v = A @ p
                if nearly_orthogonal(shadow, v):
                    return finish("breakdown", "BiCGStab: (r_shadow, Ap) численно равно нулю.")
                alpha = rho / (shadow @ v)
                s = r - alpha * v
                half_x = x + alpha * p
                # Ранний выход до вычисления omega предотвращает деление 0/0.
                half_norm = float(np.linalg.norm(b - A @ half_x))
                if half_norm <= target:
                    x = half_x
                    history.append(half_norm)
                    return finish("converged", "Допуск достигнут на первом подшаге BiCGStab.")
                t = A @ s
                if np.linalg.norm(t) == 0 or nearly_orthogonal(t, s):
                    return finish("breakdown", "BiCGStab: невозможно выполнить стабилизацию.")
                stabilization = (t @ s) / (t @ t)
                new_x = half_x + stabilization * s
                r = s - stabilization * t
                rho_old = rho

            true_r = b - A @ new_x
            norm = float(np.linalg.norm(true_r))
            if not np.isfinite(new_x).all() or not np.isfinite(norm):
                return finish("diverged", "Переполнение; возвращено последнее конечное приближение.")
            x = new_x
            history.append(norm)
            if norm <= target:
                return finish("converged", "Достигнут допуск по истинной невязке.")
            if method not in {"cg", "bicgstab"}:
                r = true_r
            if norm > 1e12 * max(history[0], target):
                return finish("diverged", "Невязка увеличилась более чем в 10^12 раз.")
    return finish("max_iter", "Достигнут предел числа итераций; требуемая точность не достигнута.")


def diagnostics(A, omega=1.05):
    """Характеристики A и спектральные радиусы матриц стационарных итераций.

    Здесь np.linalg.solve используется только для диагностических матриц B,
    а не для получения решения исследуемой СЛАУ.
    """
    A, _ = validate_system(A, np.zeros(len(A)))
    D, L, U = np.diag(np.diag(A)), np.tril(A, -1), np.triu(A, 1)
    spd = is_spd(A)
    data = dict(
        spd=spd, condition_2=float(np.linalg.cond(A)),
        dominance_margin=float(np.min(np.abs(np.diag(A))
                                     - (np.abs(A).sum(axis=1) - np.abs(np.diag(A))))),
        eigenvalues=np.linalg.eigvalsh(A) if spd else np.linalg.eigvals(A),
    )
    for name, left, right in [
        ("jacobi", D, -(L + U)),
        ("gauss_seidel", D + L, -U),
        ("sor", D + omega * L, (1 - omega) * D - omega * U),
    ]:
        try:
            B = np.linalg.solve(left, right)
            data[f"rho_{name}"] = float(np.max(np.abs(np.linalg.eigvals(B))))
        except np.linalg.LinAlgError:
            data[f"rho_{name}"] = None
    return data
