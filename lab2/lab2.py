import my_methods
import numpy as np
import matplotlib.pyplot as plt

def generate_system(n):
    A = np.array([[0.0] * n for _ in range(n)])
    b = np.array([0.0] * n)

    for i in range(n):
        for j in range(n):
            if i == j:
                A[i][j] = 1.0
            else:
                A[i][j] = 1.0 / ((i + 1) ** 2 + (j + 1))
        b[i] = 1.0 / (i + 1)

    return A, b

TOL = 1e-7

def solve_straight(A, b, ref_x, method_name, method, *args, **kwargs):
    print("Прямой метод: " + method_name)
    x = method(A, b, *args, **kwargs)
    if (np.allclose(x, ref_x, atol=TOL)):
        print("OK")
    else:
        print("Решение отличается от референсного: ", x)

def solve_iter(A, b, ref_x, method_name, method, *args, **kwargs):
    print("Итерационный метод: " + method_name)
    x, residuals = method(A, b, *args, **kwargs)
    if (np.allclose(x, ref_x, atol=TOL)):
        print("OK")
    else:
        print("Решение отличается от референсного: ", x)

    return residuals

def main():
    A, b = generate_system(12)
    ref_x = np.linalg.solve(A, b)

    print("Матрица A:")
    print(A)
    print("Столбец b:")
    print(b)
    print("Референсное решение: ", ref_x)

    solve_straight(A, b, ref_x, "Метод Гаусса с выбором главного элемента", my_methods.gauss_with_pivoting)
    solve_straight(A, b, ref_x, "LU-разложение", my_methods.solve_lu)

    iter_methods = [
        ("Метод Якоби", my_methods.solve_jacobi),
        ("Метод Зейделя", my_methods.seidel_method),
        ("Метод верхней релаксации", my_methods.sor_method),
        # ("Метод градиентного спуска", my_methods.gradient_descent), # расходится
        ("Метод минимальных невязок", my_methods.minimal_residual_method),
        ("Метод сопряженных градиентов", my_methods.conjugate_gradient),
        ("Стабилизированный метод бисопряженных градиентов", my_methods.bicgstab)
    ]

    residuals_all = []
    for iter_method in iter_methods:
        residuals_all.append((
            solve_iter(A, b, ref_x, iter_method[0], iter_method[1], tol=TOL),
            iter_method[0]
        ))

    plt.figure(figsize=(10, 6))
    for residuals in residuals_all:
        plt.semilogy(residuals[0], '-', linewidth=2, label=residuals[1])
        print(residuals[0][-1])
    plt.legend()
    plt.xlabel('Итерация')
    plt.ylabel('Норма невязки')
    plt.xticks([i for i in range(0, 1+max( [ len(a[0]) for a in residuals_all ] ))])
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

