"""Проверки алгоритмов: python -m unittest discover -s numerical_solution_of_linear_systems -v."""
import unittest

import numpy as np

from solvers import (METHOD_NAMES, diagnostics, gaussian_elimination, iterate,
                     lu_factor, lu_solve, variant_m)


class SolverTests(unittest.TestCase):
    def test_variant_coefficients(self):
        A, b = variant_m()
        for i in range(20):
            for j in range(20):
                self.assertEqual(A[i, j], {0: 10, 1: 1, 2: 0.1}.get(abs(i - j), 0))
        np.testing.assert_array_equal(b, np.arange(1, 21))

    def test_all_methods_on_variant_and_generated_spd(self):
        rng = np.random.default_rng(42)
        M = rng.normal(size=(8, 8))
        A2 = M.T @ M + 30 * np.eye(8)
        x_exact = rng.normal(size=8)
        for A, b in [variant_m(), (A2, A2 @ x_exact)]:
            reference = np.linalg.solve(A, b)
            old_A, old_b = A.copy(), b.copy()
            for direct in [gaussian_elimination, lu_solve]:
                np.testing.assert_allclose(direct(A, b), reference, atol=1e-13)
            for method in METHOD_NAMES:
                with self.subTest(method=method, size=len(A)):
                    result = iterate(A, b, method, rtol=1e-11)
                    self.assertTrue(result.converged, result.message)
                    residual = np.linalg.norm(b - A @ result.x)
                    self.assertLessEqual(residual, 1e-12 + 1e-11 * np.linalg.norm(b))
                    self.assertAlmostEqual(result.residuals[-1], residual, places=14)
                    self.assertEqual(result.residuals[0], np.linalg.norm(b))
                    np.testing.assert_allclose(result.x, reference, rtol=1e-9, atol=1e-10)
            np.testing.assert_array_equal(A, old_A)
            np.testing.assert_array_equal(b, old_b)

    def test_pivot_and_lu_applicability(self):
        A, b = np.array([[0., 2.], [1., 3.]]), np.array([4., 7.])
        np.testing.assert_allclose(gaussian_elimination(A, b), [1., 2.])
        with self.assertRaises(np.linalg.LinAlgError):
            lu_solve(A, b)
        A, _ = variant_m()
        L, U = lu_factor(A)
        np.testing.assert_allclose(L @ U, A, atol=1e-14)
        np.testing.assert_array_equal(np.diag(L), np.ones(len(A)))

    def test_singular_system(self):
        for solve in [gaussian_elimination, lu_solve]:
            with self.assertRaises(np.linalg.LinAlgError):
                solve([[1, 2], [2, 4]], [1, 2])

    def test_zero_rhs_and_exact_initial_guess(self):
        A, b = variant_m()
        for method in METHOD_NAMES:
            for rhs, x0 in [(np.zeros(20), None), (b, np.linalg.solve(A, b))]:
                result = iterate(A, rhs, method, x0=x0)
                self.assertTrue(result.converged)
                self.assertEqual(result.iterations, 0)

    def test_custom_nonsymmetric_system(self):
        A = np.array([[5., 1., 0.], [-1., 4., 1.], [0., 2., 6.]])
        exact = np.array([1., -2., 3.])
        for method in METHOD_NAMES:
            result = iterate(A, A @ exact, method)
            if method in {"cg", "steepest_descent"}:
                self.assertEqual(result.status, "not_applicable")
            else:
                self.assertTrue(result.converged, (method, result.message))
                np.testing.assert_allclose(result.x, exact, atol=1e-8)

    def test_one_step_defines_gradient_and_minimal_residual(self):
        A, b = np.array([[4., 1.], [1., 3.]]), np.array([1., 2.])
        Ar = A @ b
        sd = iterate(A, b, "steepest_descent", max_iter=1)
        mr = iterate(A, b, "minimal_residual", max_iter=1)
        np.testing.assert_allclose(sd.x, (b @ b) / (b @ Ar) * b)
        np.testing.assert_allclose(mr.x, (b @ Ar) / (Ar @ Ar) * b)
        self.assertLessEqual(mr.residuals[-1], sd.residuals[-1])

    def test_stopping_and_breakdown(self):
        A, b = variant_m()
        self.assertEqual(iterate(A, b, "jacobi", max_iter=1).status, "max_iter")
        self.assertEqual(iterate(A, b, "cg", max_iter=0).iterations, 0)
        self.assertEqual(iterate([[0., 1.], [1., 0.]], [1., 1.], "jacobi").status,
                         "not_applicable")
        self.assertEqual(iterate([[0., -1.], [1., 0.]], [1., 0.], "minimal_residual").status,
                         "breakdown")
        self.assertEqual(iterate([[1., 2.], [2., 1.]], [1., 1.], "jacobi").status,
                         "diverged")
        # SOR с omega=1 совпадает с Зейделем.
        gs = iterate(A, b, "gauss_seidel")
        sor = iterate(A, b, "sor", omega=1.)
        np.testing.assert_array_equal(gs.residuals, sor.residuals)
        # BiCGStab должен выйти до потенциального 0/0 на втором подшаге.
        result = iterate(np.eye(3), [1, 2, 3], "bicgstab")
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)

    def test_bad_input(self):
        cases = [([[1, 2]], [1]), ([[1, 0], [0, 1]], [[1], [2]]),
                 ([[np.nan]], [1]), ([[1j]], [1])]
        for A, b in cases:
            with self.assertRaises(ValueError):
                iterate(A, b, "jacobi")
        for opts in [dict(omega=2), dict(rtol=-1), dict(max_iter=-1), dict(x0=[1, 2])]:
            with self.assertRaises(ValueError):
                iterate([[1.]], [1.], "sor", **opts)

    def test_diagnostics_and_residual_monotonicity(self):
        A, b = variant_m()
        d = diagnostics(A)
        self.assertTrue(d["spd"])
        self.assertGreater(d["dominance_margin"], 0)
        for method in ["jacobi", "gauss_seidel", "sor"]:
            self.assertLess(d[f"rho_{method}"], 1)
        result = iterate(A, b, "minimal_residual")
        self.assertTrue(np.all(np.diff(result.residuals) <= 1e-13))


if __name__ == "__main__":
    unittest.main()
