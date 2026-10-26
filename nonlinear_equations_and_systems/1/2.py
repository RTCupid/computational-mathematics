from math import log

accuracy = 1e-10
initial_approximation = 0.8
minimum_derivative = 0.8


def equation(argument):
    return log(argument) + (argument - 1)**3


def derivative(argument):
    return 1 / argument + 3 * (argument - 1)**2


def bisection(left_boundary, right_boundary):
    while right_boundary - left_boundary > 2 * accuracy:
        midpoint = (left_boundary + right_boundary) / 2
        if equation(midpoint) == 0:
            return midpoint
        if equation(left_boundary) * equation(midpoint) < 0:
            right_boundary = midpoint
        else:
            left_boundary = midpoint
    return (left_boundary + right_boundary) / 2


def iterate(iteration_function):
    approximation = initial_approximation
    while abs(equation(approximation)) > minimum_derivative * accuracy:
        approximation = iteration_function(approximation)
    return approximation


fixed_derivative = derivative(initial_approximation)
iteration_methods = {
    "Простая итерация": lambda argument: argument - 0.8 * equation(argument),
    "Ньютон": lambda argument: argument - equation(argument) / derivative(argument),
    "Модифицированный Ньютон": lambda argument: argument - equation(argument) / fixed_derivative,
}

print(f"Половинное деление: {bisection(0.8, 1.2):.10f}")
for method_name, iteration_function in iteration_methods.items():
    print(f"{method_name}: {iterate(iteration_function):.10f}")
