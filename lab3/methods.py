import numpy as np

def method_bisection(f, a, b, tol=1e-6, max_iter=100):
    fa = f(a)
    fb = f(b)

    # Проверка условий применимости
    if a >= b:
        raise ValueError("Границы интервала должны удовлетворять a < b")
    if fa * fb >= 0:
        raise ValueError("Функция должна иметь разные знаки на концах интервала")

    residuals = []

    # Итерационный процесс
    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)

        # Сохранение невязки
        residuals.append(abs(fc))

        # Проверка критерия остановки
        if abs(b - a) < tol or abs(fc) < tol:
            return c, residuals

        # Выбор подынтервала для следующей итерации
        if np.sign(f(a)) * np.sign(fc) < 0:
            b = c
        else:
            a = c

    # Если превышено максимальное количество итераций
    raise RuntimeError(f"Метод не сошелся за {max_iter} итераций")

def method_chord(f, a, b, tol=1e-6, max_iter=100):
    # Проверка условий применимости
    if a >= b:
        raise ValueError("Границы интервала должны удовлетворять a < b")

    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        raise ValueError("Функция должна иметь разные знаки на концах интервала")

    # Определение неподвижного конца
    # Считаем вторую производную численно
    h = 1e-6 if tol > 1e-6 else tol
    f2a = (f(a + h) - 2*f(a) + f(a - h)) / (h**2)  # f''(a)

    if fa * f2a > 0:
        fixed_end = 'a'
    else:
        fixed_end = 'b'

    residuals = []

    # Итерационный процесс
    for i in range(max_iter):
        fa = f(a)
        fb = f(b)

        # Вычисление нового приближения
        if fixed_end == 'a':
            x = a - fa * (b - a) / (fb - fa)
        else:
            x = b - fb * (a - b) / (fa - fb)

        fx = f(x)

        # Сохранение невязки
        residuals.append(abs(fx))

        # Проверка условия сходимости
        if abs(fx) < tol:
            return x, residuals

        # Обновление интервала
        if fa * fx < 0:
            b = x
        else:
            a = x

    raise RuntimeError(f"Метод не сошелся за {max_iter} итераций")


def method_simple_iteration(F, x0, max_iter=1000, tol=1e-6, q_max=0.9):
    def G(x):
        # Преобразование F(x)=0 к виду x = x - F(x)
        # Это простейший вариант, может потребоваться другая форма для сходимости
        return x - F(x)

    x_prev = np.array(x0, dtype=float)
    n = len(x0)

    # Проверка условий сходимости
    x_test1 = x_prev.copy()
    x_test2 = x_prev + 1e-7  # небольшое возмущение

    G1 = G(x_test1)
    G2 = G(x_test2)

    # Оценка константы Липшица
    q_est = np.linalg.norm(G2 - G1) / np.linalg.norm(x_test2 - x_test1)

    if q_est >= q_max:
        raise Exception(f"Нарушено условие сходимости: q_est = {q_est:.3f} >= {q_max}")

    residuals = []

    # Итерационный процесс
    for i in range(max_iter):
        x_next = G(x_prev)

        residuals.append(np.linalg.norm(F(x_next)))

        # Проверка условия остановки
        if np.linalg.norm(x_next - x_prev) < tol:
            return x_next, residuals

        x_prev = x_next.copy()

    raise Exception(f"Метод не сошелся за {max_iter} итераций")

import methods_syslineq

def compute_jacobian(F, x, h=1e-7):
    n = len(x)
    J = np.zeros((n, n))

    for j in range(n):
        # Прямое возмущение: x + h*e_j
        x_forward = x.copy()
        x_forward[j] += h
        F_forward = F(x_forward)

        # Обратное возмущение: x - h*e_j
        x_backward = x.copy()
        x_backward[j] -= h
        F_backward = F(x_backward)

        for i in range(n):
            J[i, j] = (F_forward[i] - F_backward[i]) / (2 * h)

    return J

def method_newton(F, x0, tol=1e-8, max_iter=100):
    x = np.array(x0, dtype=float)

    # Проверка размерностей
    if len(x) != len(F(x)):
        raise ValueError("Размерность вектора и системы уравнений не совпадают")

    residuals = []

    for i in range(max_iter):
        F_val = F(x)
        J_val = compute_jacobian(F, x)

        # Проверка вырожденности матрицы Якоби
        if np.linalg.det(J_val) == 0:
            raise ValueError("Матрица Якоби вырождена")

        # Решение системы J * dx = -F
        dx = methods_syslineq.solve_lu(J_val, -F_val)

        # Обновление решения
        x += dx

        residuals.append(np.linalg.norm(F(x)))

        # Проверка сходимости
        if np.linalg.norm(dx) < tol:
            return x, residuals

    raise RuntimeError("Метод не сошелся за максимальное число итераций")


def method_modified_newton(F, x0, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)

    # Проверка размерностей
    if len(x) != len(F(x)):
        raise ValueError("Размерность вектора и системы уравнений не совпадают")

    # Вычисляем Якобиан один раз в начальной точке
    J0 = compute_jacobian(F, x0)

    # Проверка обратимости матрицы Якоби
    if np.abs(np.linalg.det(J0)) < 1e-12:
        raise ValueError("Матрица Якоби в начальной точке вырождена")

    residuals = []

    # Итерационный процесс
    for iter_num in range(max_iter):
        F_current = F(x)

        residuals.append(np.linalg.norm(F_current))

        # Проверка достижения точности
        if np.linalg.norm(F_current) < tol:
            return x, residuals

        dx = methods_syslineq.solve_lu(J0, -F_current)

        # Обновление решения
        x += dx

        # Дополнительная проверка сходимости
        if np.linalg.norm(dx) < tol:
            return x, residuals

    raise Exception(f"Метод не сошелся за {max_iter} итераций")

