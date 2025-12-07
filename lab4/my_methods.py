import numpy as np
import bisect
from typing import List, Callable, Union, Tuple, Optional
from numbers import Number

import methods_syslineq


# =============== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===============

def _validate_input_data(
    x_points: List[float],
    y_points: List[float],
    check_increasing: bool = False,
    min_points: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Проверяет и подготавливает входные данные для интерполяции.

    Parameters
    ----------
    x_points : List[float]
        Список координат x.
    y_points : List[float]
        Список значений функции.
    check_increasing : bool, optional
        Проверять строгое возрастание x (по умолчанию False).
    min_points : int, optional
        Минимальное количество точек (по умолчанию 1).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Проверенные и преобразованные массивы x и y.

    Raises
    ------
    ValueError
        При некорректных входных данных.
    """
    if len(x_points) != len(y_points):
        raise ValueError("Количество узлов и значений функции должно совпадать")

    if len(x_points) < min_points:
        raise ValueError(f"Нужно хотя бы {min_points} точка(ы) для интерполяции")

    x = np.asarray(x_points, dtype=float)
    y = np.asarray(y_points, dtype=float)

    if check_increasing and not np.all(np.diff(x) > 0):
        raise ValueError("Узлы x должны быть строго возрастающими")

    return x, y


def _evaluate_poly_horner(
    x_val: Union[float, np.ndarray, List[float]],
    coeffs: np.ndarray,
    base_points: Optional[np.ndarray] = None
) -> Union[float, np.ndarray]:
    """
    Вычисляет значение полинома по схеме Горнера.

    Parameters
    ----------
    x_val : Union[float, np.ndarray, List[float]]
        Точка или массив точек для вычисления.
    coeffs : np.ndarray
        Коэффициенты полинома.
    base_points : Optional[np.ndarray]
        Базисные точки для полинома Ньютона (x_i в множителях (x-x_i)).

    Returns
    -------
    Union[float, np.ndarray]
        Значение полинома в точке(ах).
    """
    # Преобразуем входные данные к numpy массиву
    x_arr = np.asarray(x_val, dtype=float)
    is_scalar = x_arr.ndim == 0

    if is_scalar:
        x_arr = x_arr.reshape(1)

    result = np.full_like(x_arr, coeffs[-1])

    if base_points is None:
        # Полином в стандартной форме: c0 + c1*x + c2*x² + ...
        for i in range(len(coeffs) - 2, -1, -1):
            result = result * x_arr + coeffs[i]
    else:
        # Полином Ньютона: c0 + c1*(x-x0) + c2*(x-x0)*(x-x1) + ...
        n = len(coeffs) - 1
        for i in range(n - 1, -1, -1):
            result = result * (x_arr - base_points[i]) + coeffs[i]

    return result[0] if is_scalar else result


def _tridiagonal_solve(
    a: List[float],
    b: List[float],
    c: List[float],
    d: List[float]
) -> List[float]:
    """
    Решает трёхдиагональную систему методом прогонки (Томаса).

    Система: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]

    Parameters
    ----------
    a, b, c : List[float]
        Коэффициенты трёхдиагональной матрицы.
    d : List[float]
        Вектор правой части.

    Returns
    -------
    List[float]
        Решение системы.

    Raises
    ------
    ValueError
        При вырожденной системе.
    """
    n = len(d)

    if n == 0:
        return []

    # Прямой ход
    alpha = [0.0] * n
    beta = [0.0] * n

    # Начальные значения
    if b[0] == 0:
        raise ValueError("Вырожденная система: b[0] равно 0")

    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * alpha[i-1]
        if denom == 0:
            raise ValueError("Вырожденная система")
        alpha[i] = -c[i] / denom
        beta[i] = (d[i] - a[i] * beta[i-1]) / denom

    # Обратный ход
    x = [0.0] * n
    x[n-1] = beta[n-1]

    for i in range(n-2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]

    return x


def _compute_divided_differences(
    x: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Вычисляет разделённые разности для интерполяции Ньютона.

    Parameters
    ----------
    x : np.ndarray
        Узлы интерполяции.
    y : np.ndarray
        Значения функции в узлах.

    Returns
    -------
    np.ndarray
        Коэффициенты полинома Ньютона.
    """
    n = len(x) - 1
    coeffs = y.copy()

    for k in range(1, n + 1):
        for i in range(n, k - 1, -1):
            coeffs[i] = (coeffs[i] - coeffs[i-1]) / (x[i] - x[i-k])

    return coeffs


# =============== ОСНОВНЫЕ ФУНКЦИИ ===============

def newton_interpolation(
    x_points: List[float],
    f_values: List[float]
) -> Callable[[Union[float, List, np.ndarray]], Union[float, np.ndarray]]:
    """
    Строит интерполяционный полином Ньютона по заданным узлам.

    Parameters
    ----------
    x_points : List[float]
        Список узлов интерполяции (x_i).
    f_values : List[float]
        Список значений функции в узлах (f_i).

    Returns
    -------
    Callable
        Функция, вычисляющая значение полинома Ньютона в точке x.
        Поддерживает скалярные значения, списки и numpy массивы.

    Examples
    --------
    >>> f = newton_interpolation([0, 1, 2], [1, 2, 4])
    >>> f(0.5)
    1.5
    """
    x, f = _validate_input_data(x_points, f_values, check_increasing=True, min_points=2)

    # Вычисление коэффициентов полинома Ньютона
    coeffs = _compute_divided_differences(x, f)

    def polynomial(x_val: Union[float, List, np.ndarray]) -> Union[float, np.ndarray]:
        return _evaluate_poly_horner(x_val, coeffs, x[:len(coeffs)-1])

    return polynomial


def cubic_spline_interpolation(
    x: List[float],
    y: List[float]
) -> Callable[[Union[float, List, np.ndarray]], Union[float, np.ndarray]]:
    """
    Построение кубического сплайна с естественными граничными условиями.

    Parameters
    ----------
    x : List[float]
        Список узлов интерполяции (должны быть строго возрастающими).
    y : List[float]
        Значения функции в узлах.

    Returns
    -------
    Callable[[Union[float, List, np.ndarray]], Union[float, np.ndarray]]
        Функция-сплайн, вычисляющая значение в точках.
        Поддерживает скалярные значения, списки и numpy массивы.
        При включенной экстраполяции работает для любых x.

    Raises
    ------
    ValueError
        Если входные данные некорректны или при extrapolate=False и x вне интервала.
    """
    x_arr, y_arr = _validate_input_data(x, y, check_increasing=True, min_points=2)
    n = len(x_arr) - 1

    # Шаг 1: Вычисление разностей
    h = [x_arr[i+1] - x_arr[i] for i in range(n)]
    delta = [(y_arr[i+1] - y_arr[i]) / h[i] for i in range(n)]

    # Шаг 2: Подготовка системы для моментов M
    # Естественные граничные условия: M[0] = M[n] = 0
    mu = [0.0] * (n)      # μ_i для i=1..n-1
    lam = [0.0] * (n)     # λ_i для i=1..n-1
    d = [0.0] * (n+1)     # правые части

    for i in range(1, n):
        mu[i] = h[i-1] / (h[i-1] + h[i])
        lam[i] = 1 - mu[i]
        d[i] = 6 * (delta[i] - delta[i-1]) / (h[i-1] + h[i])

    # Шаг 3: Решение трёхдиагональной системы для M[1]..M[n-1]
    if n > 1:
        # Подготовка коэффициентов для метода прогонки
        a_coeffs = [0.0] + mu[2:n]  # a[0] не используется
        b_coeffs = [2.0] * (n-1)
        c_coeffs = lam[1:n]
        d_coeffs = d[1:n]

        M_inner = _tridiagonal_solve(a_coeffs, b_coeffs, c_coeffs, d_coeffs)

        # Сборка полного вектора M
        M = [0.0] * (n+1)
        for i in range(1, n):
            M[i] = M_inner[i-1]
    else:
        M = [0.0, 0.0]

    # Шаг 4: Вычисление коэффициентов сплайна для каждого отрезка
    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    d_spline = [0.0] * n  # коэффициенты сплайна

    for i in range(n):
        a[i] = y_arr[i]
        c[i] = M[i] / 2.0
        d_spline[i] = (M[i+1] - M[i]) / (6.0 * h[i])
        b[i] = delta[i] - h[i] * (2.0 * M[i] + M[i+1]) / 6.0

    # Преобразуем списки в numpy массивы для удобства индексации
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    c_arr = np.array(c, dtype=float)
    d_arr = np.array(d_spline, dtype=float)
    x_arr = np.array(x_arr, dtype=float)

    # Шаг 5: Создание интерполирующей функции с поддержкой массивов
    def spline_eval(points: Union[float, List, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Вычисляет значение сплайна в заданных точках.

        Parameters
        ----------
        points : Union[float, List, np.ndarray]
            Точки для вычисления значений сплайна.

        Returns
        -------
        Union[float, np.ndarray]
            Значения сплайна в заданных точках.
        """
        # Преобразуем вход в numpy массив
        points_arr = np.asarray(points, dtype=float)
        is_scalar = points_arr.ndim == 0

        if is_scalar:
            points_arr = points_arr.reshape(1)

        # Находим индексы отрезков для всех точек
        # Используем searchsorted для векторизованного поиска
        indices = np.searchsorted(x_arr, points_arr, side='right') - 1

        # Для экстраполяции слева: индекс 0
        indices = np.where(points_arr < x_arr[0], 0, indices)
        # Для экстраполяции справа: индекс n-1
        indices = np.where(points_arr > x_arr[n], n-1, indices)

        # Вычисляем значения сплайна для всех точек
        dx = points_arr - x_arr[indices]
        result = a_arr[indices] + b_arr[indices] * dx + c_arr[indices] * dx**2 + d_arr[indices] * dx**3

        # Возвращаем результат в исходном формате
        return result[0] if is_scalar else result

    return spline_eval

def least_squares_poly(
    x_data: List[Union[float, int]],
    y_data: List[Union[float, int]],
    degree: int = 1
) -> Callable[[Union[float, List, np.ndarray]], Union[float, np.ndarray]]:
    """
    Строит аппроксимирующий полином заданной степени методом наименьших квадратов.

    Parameters
    ----------
    x_data : List[Union[float, int]]
        Список координат x узловых точек.
    y_data : List[Union[float, int]]
        Список значений функции в узловых точках (y_i = f(x_i)).
    degree : int, optional
        Степень аппроксимирующего полинома. Должна быть неотрицательной.
        По умолчанию 1.

    Returns
    -------
    Callable
        Функция-полином P(x), вычисляющая значение аппроксимации в точке (точках) x.
        Поддерживает скалярные значения, списки и numpy массивы.

    Raises
    ------
    ValueError
        Если входные данные некорректны.

    Examples
    --------
    >>> f = least_squares_poly([0, 1, 2], [1, 2, 4], degree=1)
    >>> f(0.5)
    1.5
    """
    if degree < 0:
        raise ValueError("Степень полинома должна быть неотрицательной")

    x_arr, y_arr = _validate_input_data(x_data, y_data, min_points=degree+1)

    if len(x_arr) <= degree:
        raise ValueError(
            f"Недостаточно данных. Требуется как минимум {degree + 1} точка "
            f"для полинома степени {degree}"
        )

    n = len(x_arr)
    m = degree + 1  # количество коэффициентов

    # Построение матрицы Вандермонда с использованием векторизации
    A = np.zeros((n, m))
    for j in range(m):
        A[:, j] = x_arr ** j

    # Вычисление системы нормальных уравнений
    # ATA = A.T @ A
    # ATy = A.T @ y_arr
    ATA = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            sum_val = 0.0
            for k in range(n):
                sum_val += A[k, i] * A[k, j]
            ATA[i, j] = sum_val

    # Вычисляем A^T * y
    ATy = np.zeros(m, dtype=float)
    for i in range(m):
        sum_val = 0.0
        for k in range(n):
            sum_val += A[k, i] * y_arr[k]
        ATy[i] = sum_val

    coeffs = np.linalg.solve(ATA, ATy)

    def poly_func(x_input: Union[float, List, np.ndarray]) -> Union[float, np.ndarray]:
        return _evaluate_poly_horner(x_input, coeffs)

    return poly_func
