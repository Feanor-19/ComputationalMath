import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify

def create_functions_from_expressions(expressions, x_values):
    """
    Создает функции и их производные из строковых выражений

    Параметры:
    expressions: list of str
        Список строковых представлений функций
    x_values: list of float
        Список точек x_i

    Возвращает:
    f_list, f_der_list: списки функций и их производных
    """
    f_list = []
    f_der_list = []
    func_names = []

    x_sym = sp.Symbol('x')

    for expr_str in expressions:
        expr = sp.sympify(expr_str)

        f_lambda = lambdify(x_sym, expr, 'numpy')
        f_list.append(f_lambda)

        expr_der = sp.diff(expr, x_sym)
        f_der_lambda = lambdify(x_sym, expr_der, 'numpy')
        f_der_list.append(f_der_lambda)

        func_names.append(expr_str)

    return f_list, f_der_list, func_names

def plot_errors(expressions, x_list, F_list):
    """
    Построение графиков абсолютных ошибок для функций, заданных строками

    Параметры:
    expressions: list of str
        Список строковых представлений функций f_i(x)
    x_list: list of float
        Список точек x_i
    F_list: list of callable
        Список функций F_j(f, x, h)
    """
    f_list, f_der_list, func_names = create_functions_from_expressions(expressions, x_list)

    n_values = np.arange(1, 22)
    h_values = 2 / (2 ** n_values)

    for i, (f, f_der, x, func_name) in enumerate(zip(f_list, f_der_list, x_list, func_names)):
        plt.figure(i + 1, figsize=(10, 6))
        exact_value = f_der(x)

        for j, F in enumerate(F_list):
            errors = []
            for h in h_values:
                try:
                    approx = F(f, x, h)
                    error = abs(approx - exact_value)
                    errors.append(error)
                except Exception as e:
                    print(f"Ошибка при вычислении F_{j}(f_{i}, x={x}, h={h}): {e}")
                    errors.append(1e-16)

            print(errors)
            plt.plot(n_values, errors, 'o-', label=f'F_{j+1}', markersize=4)

        plt.title(f'Абсолютная ошибка для функции: {func_name} в точке x = {x}')
        plt.ylabel('Абсолютная ошибка')
        plt.yscale('log')
        plt.xlabel('n')
        plt.xticks(n_values)

        plt.grid(True, which='major', axis='both', alpha=0.3)
        plt.grid(True, which='minor', axis='y', alpha=0.2)

        plt.legend()

        plt.show()

if __name__ == "__main__":
    expressions = [
        "sin(x**2)",
        "cos(sin(x))",
        "exp(sin(cos(x)))",
        "log(x + 3)",
        "(x+3)**(0.5)"
    ]

    x_list = [2.0, np.pi/4, 1.0, 0.5, 0.5]

    F_list = [
        lambda f, x, h: (f(x + h) - f(x)) / h,
        lambda f, x, h: (f(x) - f(x - h)) / h,
        lambda f, x, h: (f(x + h) - f(x - h)) / (2*h),
        lambda f, x, h: (4/3)*((f(x + h) - f(x - h)) / (2*h)) - (1/3)*((f(x + 2*h) - f(x - 2*h)) / (4*h)),
        lambda f, x, h: (3/2)*((f(x + h) - f(x - h)) / (2*h)) - (3/5)*((f(x + 2*h) - f(x - 2*h)) / (4*h)) \
            + (1/10)*((f(x + 3*h) - f(x - 3*h)) / (6*h))
    ]

    plot_errors(expressions, x_list, F_list)
