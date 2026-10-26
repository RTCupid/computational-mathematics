from math import cos, sin

accuracy = 1e-10
initial_approximation = 3.0
minimum_derivative = 20.0


def equation(argument):
    return argument**2 - 20 * sin(argument)


def derivative(argument):
    return 2 * argument - 20 * cos(argument)


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
    "Простая итерация": lambda argument: argument - equation(argument) / 25,
    "Ньютон": lambda argument: argument - equation(argument) / derivative(argument),
    "Модифицированный Ньютон": lambda argument: argument - equation(argument) / fixed_derivative,
}

print("Точный корень: 0")
print(f"Половинное деление: {bisection(2.5, 3.0):.10f}")
for method_name, iteration_function in iteration_methods.items():
    print(f"{method_name}: {iterate(iteration_function):.10f}")
