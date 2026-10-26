from math import cos, sin

accuracy = 1e-10
initial_approximation = (0.5, -1.0)
maximum_iterations = 1000


def equation(x, y):
    return sin(x + 2) - y - 1.5, x + cos(x - 2) - 0.5


def jacobian(x, y):
    return (cos(x + 2), -1), (1 - sin(x - 2), 0)


def ordinate(x):
    return sin(x + 2) - 1.5


def bisection(left_boundary, right_boundary):
    while right_boundary - left_boundary > 2 * accuracy:
        midpoint = (left_boundary + right_boundary) / 2
        midpoint_value = equation(midpoint, ordinate(midpoint))[1]
        if midpoint_value == 0:
            return midpoint, ordinate(midpoint)
        left_value = equation(left_boundary, ordinate(left_boundary))[1]
        if left_value * midpoint_value < 0:
            right_boundary = midpoint
        else:
            left_boundary = midpoint
    x = (left_boundary + right_boundary) / 2
    return x, ordinate(x)


def simple_iteration(x, y):
    return (x + 0.5 - cos(x - 2)) / 2, ordinate(x)


def newton_step(x, y, matrix):
    first_value, second_value = equation(x, y)
    (a, b), (c, d) = matrix
    determinant = a * d - b * c
    return (
        x - (d * first_value - b * second_value) / determinant,
        y - (a * second_value - c * first_value) / determinant,
    )


def iterate(iteration_function):
    approximation = initial_approximation
    for _ in range(maximum_iterations):
        next_approximation = iteration_function(*approximation)
        step = max(abs(new - old) for new, old in zip(next_approximation, approximation))
        residual = max(abs(value) for value in equation(*next_approximation))
        if step <= accuracy and residual <= accuracy:
            return next_approximation
        approximation = next_approximation
    raise RuntimeError("Метод не сошёлся за 1000 итераций")


fixed_jacobian = jacobian(*initial_approximation)
iteration_methods = {
    "Простая итерация": simple_iteration,
    "Ньютон": lambda x, y: newton_step(x, y, jacobian(x, y)),
    "Модифицированный Ньютон": lambda x, y: newton_step(x, y, fixed_jacobian),
}

x, y = bisection(0.0, 1.0)
print(f"Половинное деление: x = {x:.10f}, y = {y:.10f}")
for method_name, iteration_function in iteration_methods.items():
    x, y = iterate(iteration_function)
    print(f"{method_name}: x = {x:.10f}, y = {y:.10f}")
