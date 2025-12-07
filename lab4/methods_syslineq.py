# copied from lab2

def gauss_with_pivoting(A, b):
    """
    Решение СЛАУ Ax=b методом Гаусса с выбором главного элемента
    """
    n = len(A)
    A = [row[:] for row in A]
    b = b[:]

    # Прямой ход с выбором главного элемента
    for k in range(n):
        # Поиск главного элемента в столбце k
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[max_row][k]):
                max_row = i

        # Перестановка строк
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]

        # Исключение
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    # Обратный ход
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]

    return x

import math
import numpy as np

def solve_lu(A, b):
    """
    Решение СЛАУ Ax=b методом LU-разложения
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = A.shape[0]

    # Проверка условий применимости
    if A.shape[1] != n:
        raise ValueError("Матрица A должна быть квадратной")
    if b.shape[0] != n:
        raise ValueError("Размеры A и b не совпадают")
    if np.linalg.det(A) == 0:
        raise ArithmeticError("Матрица A вырождена")

    # Инициализация матриц L и U
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # LU-разложение (алгоритм Краута)
    for i in range(n):
        # Вычисление элементов U
        for k in range(i, n):
            sum_val = 0.0
            for j in range(i):
                sum_val += L[i, j] * U[j, k]
            U[i, k] = A[i, k] - sum_val

        # Проверка возможности разложения
        if abs(U[i, i]) < 1e-12:
            raise ArithmeticError("Нулевой диагональный элемент. LU-разложение невозможно")

        # Вычисление элементов L
        for k in range(i, n):
            if i == k:
                L[i, i] = 1.0
            else:
                sum_val = 0.0
                for j in range(i):
                    sum_val += L[k, j] * U[j, i]
                L[k, i] = (A[k, i] - sum_val) / U[i, i]

    # Решение системы Ly = b прямой подстановкой
    y = np.zeros(n)
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L[i, j] * y[j]
        y[i] = b[i] - sum_val

    # Решение системы Ux = y обратной подстановкой
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_val = 0.0
        for j in range(i+1, n):
            sum_val += U[i, j] * x[j]
        x[i] = (y[i] - sum_val) / U[i, i]

    return x

def compute_residual_norm(x, n):
    res = 0
    for i in range(n):
        res += x[i]**2
    return math.sqrt(res)

def solve_jacobi(A, b, max_iter=1000, tol=1e-10):
    """
    Решает СЛАУ Ax=b методом Якоби
    """
    n = len(A)
    x = np.array([0.0] * n)  # Начальное приближение
    x_new = np.array([0.0] * n)
    residuals = []  # Массив для хранения норм невязок

    for iteration in range(max_iter):
        # Вычисление нового приближения
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x[j]
            x_new[i] = (b[i] - s) / A[i][i]

        # Вычисление нормы невязки для текущего приближения x_new
        residual_norm = 0.0
        residual = []
        for i in range(n):
            # Вычисление i-й компоненты невязки: r_i = b_i - sum(A_ij * x_j)
            r_i = b[i]
            for j in range(n):
                r_i -= A[i][j] * x_new[j]
            residual.append(r_i)
        residual_norm = compute_residual_norm(residual, n)
        residuals.append(residual_norm)

        # Проверка сходимости
        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            return x_new, residuals

        x = x_new.copy()

    raise RuntimeError("Достигнуто максимальное число итераций")

def seidel_method(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Решение СЛАУ методом Зейделя
    """
    n = A.shape[0]

    # Проверка условий применимости
    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица A должна быть квадратной")

    if A.shape[0] != b.shape[0]:
        raise ValueError("Размерности A и b не согласованы")

    # Инициализация
    x = x0 if x0 is not None else np.zeros(n, dtype=float)
    x_new = x.copy()
    residuals = []

    for iteration in range(max_iter):
        max_diff = 0.0

        # Последовательное обновление компонент
        for i in range(n):
            sum1 = 0.0
            for j in range(i):
                sum1 += A[i, j] * x_new[j]

            sum2 = 0.0
            for j in range(i + 1, n):
                sum2 += A[i, j] * x[j]

            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

            # Вычисление максимального изменения
            diff = abs(x_new[i] - x[i])
            if diff > max_diff:
                max_diff = diff

        # Вычисление невязки
        residual = 0.0
        for i in range(n):
            Ax_i = 0.0
            for j in range(n):
                Ax_i += A[i, j] * x_new[j]
            residual += (b[i] - Ax_i) ** 2
        residual = np.sqrt(residual)
        residuals.append(residual)

        # Проверка критерия остановки
        if max_diff < tol:
            return x_new, np.array(residuals)

        x = x_new.copy()

    raise ValueError(f"Метод не сошелся за {max_iter} итераций")


def sor_method(A, b, omega=1.1, tol=1e-10, max_iter=1000):
    """
    Решает СЛАУ Ax = b методом верхней релаксации (SOR).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(A)
    x = np.zeros(n)
    divergence_count = 0
    prev_norm = float('inf')
    residuals = []  # Массив для хранения норм невязок

    for iteration in range(max_iter):
        x_old = x.copy()
        max_diff = 0.0

        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]

            new_xi = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - sigma)
            max_diff = max(max_diff, abs(new_xi - x_old[i]))
            x[i] = new_xi

        # Вычисление нормы невязки
        residual_norm = 0.0
        for i in range(n):
            Ax_i = 0.0
            for j in range(n):
                Ax_i += A[i, j] * x[j]
            residual = b[i] - Ax_i
            residual_norm += residual * residual

        residual_norm = np.sqrt(residual_norm)
        residuals.append(residual_norm)

        # Контроль расходимости
        if residual_norm > prev_norm * 1.5:  # Если норма невязки растет
            divergence_count += 1
            if divergence_count > 5:
                raise RuntimeError(f"Метод расходится после {iteration+1} итераций. "
                                 f"Попробуйте другой параметр omega или метод.")
        else:
            divergence_count = 0

        prev_norm = residual_norm

        if max_diff < tol:
            return x, np.array(residuals)

        # Защита от переполнения
        if np.any(np.abs(x) > 1e100):
            raise RuntimeError("Переполнение: значения решения стали слишком большими. "
                             "Метод расходится.")

    raise RuntimeError(f"Метод не сошелся за {max_iter} итераций. "
                      f"Последняя достигнутая точность: {max_diff:.2e}")

def gradient_descent(A, b, max_iter=1000, tol=1e-10, x0=None):
    """
    Решение СЛАУ Ax=b методом градиентного спуска
    """
    n = len(b)

    # Проверка условий применимости
    if A.shape != (n, n):
        raise ValueError("Матрица A должна быть квадратной и соответствовать размеру b")

    # Инициализация
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Массив для хранения норм невязок
    residuals = []

    for iter in range(max_iter):
        # Вычисление невязки r = Ax - b (покомпонентно)
        r = np.zeros(n)
        for i in range(n):
            for j in range(n):
                r[i] += A[i, j] * x[j]
            r[i] -= b[i]

        # Вычисление нормы невязки
        r_norm = compute_residual_norm(r, n)

        # Сохранение нормы невязки
        residuals.append(r_norm)

        # Проверка условия останова
        if r_norm < tol:
            break

        # Вычисление Ar
        Ar = np.zeros(n)
        for i in range(n):
            for j in range(n):
                Ar[i] += A[i, j] * r[j]

        # Вычисление шага alpha
        numerator = 0.0
        denominator = 0.0
        for i in range(n):
            numerator += r[i] * r[i]
            denominator += r[i] * Ar[i]

        if abs(denominator) < 1e-12:
            raise ZeroDivisionError("Деление на ноль при вычислении шага")

        alpha = numerator / denominator

        # Обновление решения
        for i in range(n):
            x[i] -= alpha * r[i]

    return x, residuals

def minimal_residual_method(A, b, x0=None, max_iter=1000, tol=1e-10):
    """
    Решает СЛАУ Ax=b методом минимальных невязок.
    """
    n = A.shape[0]

    # Проверка условий применимости
    if A.shape != (n, n):
        raise ValueError("Матрица A должна быть квадратной")
    if b.shape != (n,):
        raise ValueError("Размерность b не соответствует A")

    # Инициализация
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Массив для хранения норм невязок
    residual_norms = []

    # Основной итерационный процесс
    for k in range(max_iter):
        # Вычисление невязки r = b - A*x
        r = np.zeros(n)
        for i in range(n):
            r[i] = b[i]
            for j in range(n):
                r[i] -= A[i, j] * x[j]

        # Вычисление и сохранение нормы невязки
        r_norm = compute_residual_norm(r, n)
        residual_norms.append(r_norm)

        # Проверка критерия остановки
        if r_norm < tol:
            break

        # Вычисление Ar
        Ar = np.zeros(n)
        for i in range(n):
            for j in range(n):
                Ar[i] += A[i, j] * r[j]

        # Вычисление итерационного параметра τ
        numerator = 0.0
        denominator = 0.0
        for i in range(n):
            numerator += r[i] * Ar[i]
            denominator += Ar[i] * Ar[i]

        if abs(denominator) < 1e-12:
            break

        tau = numerator / denominator

        # Обновление решения
        for i in range(n):
            x[i] += tau * r[i]

    return x, residual_norms

def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Решает СЛАУ Ax = b методом сопряжённых градиентов.
    """
    n = len(A)

    # Проверка условий применимости
    if any(len(row) != n for row in A):
        raise ValueError("Матрица A должна быть квадратной")

    # Инициализация
    if x0 is None:
        x = [0.0] * n
    else:
        x = x0.copy()

    # Вычисление начальной невязки r0 = b - Ax0
    r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    p = r.copy()
    rs_old = sum(r_i * r_i for r_i in r)

    # Массив для хранения норм невязок
    residuals = [rs_old ** 0.5]  # Норма начальной невязки

    for k in range(max_iter):
        # Вычисление Ap
        Ap = [sum(A[i][j] * p[j] for j in range(n)) for i in range(n)]

        # Вычисление шага alpha
        pAp = sum(p[i] * Ap[i] for i in range(n))
        if abs(pAp) < 1e-15:
            break
        alpha = rs_old / pAp

        # Обновление решения и невязки
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]

        # Вычисление новой нормы невязки
        rs_new = sum(r_i * r_i for r_i in r)
        residual_norm = compute_residual_norm(r, n)
        residuals.append(residual_norm)

        # Проверка сходимости
        if residual_norm < tol:
            break

        # Вычисление beta и нового направления
        beta = rs_new / rs_old
        for i in range(n):
            p[i] = r[i] + beta * p[i]
        rs_old = rs_new

    return x, residuals

def bicgstab(A, b, x0=None, tol=1e-10, max_iter=1000, r0_star=None):
    """
    Стабилизированный метод бисопряженных градиентов (BiCGSTAB) для решения СЛАУ Ax = b
    """
    # Преобразуем входные данные в numpy массивы, если они еще не являются таковыми
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = A.shape[0]

    # Проверка условий применимости
    if A.shape[1] != n:
        raise ValueError("Матрица A должна быть квадратной")

    if n != len(b):
        raise ValueError("Размеры матрицы A и вектора b не совместимы")

    # Инициализация
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x0 = np.array(x0, dtype=float)
        if len(x0) != n:
            raise ValueError("Начальное приближение x0 должно иметь размер n")
        x = x0.copy()

    # Вспомогательные функции
    def matvec(A, x):
        """Умножение матрицы на вектор"""
        n = A.shape[0]
        result = np.zeros(n, dtype=float)
        for i in range(n):
            for j in range(n):
                result[i] += A[i, j] * x[j]
        return result

    def dot(u, v):
        """Скалярное произведение"""
        result = 0.0
        for i in range(len(u)):
            result += u[i] * v[i]
        return result

    def vector_add(u, v, alpha=1.0, beta=1.0):
        """Линейная комбинация векторов"""
        n = len(u)
        result = np.zeros(n, dtype=float)
        for i in range(n):
            result[i] = alpha * u[i] + beta * v[i]
        return result

    # Вычисление начальной невязки
    r = vector_add(b, matvec(A, x), alpha=1.0, beta=-1.0)

    # Массив для хранения норм невязок
    residual_norms = []
    initial_residual_norm = compute_residual_norm(r, n)
    residual_norms.append(initial_residual_norm)

    # Выбор вспомогательного вектора
    if r0_star is None:
        r0_star = r.copy()
    else:
        r0_star = np.array(r0_star, dtype=float)
        if len(r0_star) != n:
            raise ValueError("Вспомогательный вектор r0_star должен иметь размер n")

    rho_prev = 1.0
    alpha = 1.0
    omega = 1.0
    p = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)

    # Основной цикл
    for k in range(max_iter):
        rho = dot(r, r0_star)

        # Проверка breakdown
        if abs(rho) < 1e-16:
            raise ArithmeticError("Breakdown: rho = 0")

        if k == 0:
            p = r.copy()
        else:
            beta = (rho / rho_prev) * (alpha / omega)
            # p = r + beta * (p - omega * v)
            p_temp = vector_add(p, v, alpha=1.0, beta=-omega)
            p = vector_add(r, p_temp, alpha=1.0, beta=beta)

        v = matvec(A, p)

        alpha_denom = dot(v, r0_star)
        if abs(alpha_denom) < 1e-16:
            raise ArithmeticError("Breakdown: (v, r0_star) = 0")

        alpha = rho / alpha_denom

        s = vector_add(r, v, alpha=1.0, beta=-alpha)

        # Проверка сходимости по s
        s_norm = np.sqrt(dot(s, s))
        if s_norm < tol:
            x = vector_add(x, p, alpha=1.0, beta=alpha)
            # Вычисляем финальную невязку
            r_final = vector_add(b, matvec(A, x), alpha=1.0, beta=-1.0)
            final_residual_norm = compute_residual_norm(r_final, n)
            residual_norms.append(final_residual_norm)
            return x, np.array(residual_norms)

        t = matvec(A, s)

        omega_denom = dot(t, t)
        if abs(omega_denom) < 1e-16:
            raise ArithmeticError("Breakdown: (t, t) = 0")

        omega = dot(t, s) / omega_denom

        # Обновление решения и невязки
        # x = x + alpha * p + omega * s
        x_temp = vector_add(p, s, alpha=alpha, beta=omega)
        x = vector_add(x, x_temp, alpha=1.0, beta=1.0)

        r = vector_add(s, t, alpha=1.0, beta=-omega)

        # Сохраняем норму невязки на текущей итерации
        r_norm = compute_residual_norm(r, n)
        residual_norms.append(r_norm)

        # Проверка сходимости
        if r_norm < tol:
            return x, np.array(residual_norms)

        rho_prev = rho

    raise RuntimeError(f"Метод не сошелся за {max_iter} итераций")

