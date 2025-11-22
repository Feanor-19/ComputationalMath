import copy
import numpy as np
import my_methods

def solve_placeholder(A, b):
    return np.linalg.solve(A, b)

def read_test_data(filename):
    tests = []
    with open(filename, 'r') as f:
        content = f.read().strip().split('---\n')
        for block in content:
            lines = block.strip().split('\n')
            N = int(lines[0])

            # Чтение матрицы A
            A = []
            for i in range(1, 1 + N):
                A.append(list(map(float, lines[i].split())))

            # Чтение вектора b
            b = list(map(float, lines[1 + N].split()))

            # Чтение эталонного решения x
            x_true = list(map(float, lines[2 + N].split()))

            tests.append((N, np.array(A), np.array(b), np.array(x_true)))
    return tests

def check_solution(x_true, x_calc, acceptable_error):
    error = np.linalg.norm(x_calc - x_true) / np.linalg.norm(x_true)
    return error <= acceptable_error, error

def run_tests(tests, method_name, solve_function, is_iter, *args, **kwargs):
    """
    Прогоняет набор тестов для заданного метода решения.

    Args:
        tests: список тестов (N, A, b, x_true)
        method_name: имя метода для вывода
        solve_function: функция решения СЛАУ (A, b, *args, **kwargs) -> x
        *args, **kwargs: дополнительные аргументы для solve_function
    """
    print(f"\n--- Тестирование метода: {method_name} ---")

    passed = 0

    for i, (N, A, b, x_true) in enumerate(tests, 1):
        try:
            A_copy = copy.deepcopy(A)
            b_copy = copy.deepcopy(b)
            if is_iter:
                x_calc, _ = solve_function(A_copy, b_copy, *args, **kwargs)
            else:
                x_calc = solve_function(A_copy, b_copy, *args, **kwargs)
            is_correct, error = check_solution(x_true, x_calc, 0.01)

            if is_correct:
                print(f"Тест {i}: OK (погрешность: {error:.6f})")
                passed += 1
            else:
                print(f"Тест {i}: Ошибка. Относительная погрешность: {error:.6f}")
        except Exception as e:
            print(f"Тест {i}: Ошибка метода - {str(e)}")

    print(f"Успешно пройдено тестов: {passed}/{len(tests)}")

def main():
    tests_simple = read_test_data('lab2\\checker_tests\\simple.txt')
    tests_random = read_test_data('lab2\\checker_tests\\rand.txt')
    tests_random_sym = read_test_data('lab2\\checker_tests\\rand_sym.txt')


    tests_all = tests_simple + tests_random
    tests_all_sym = tests_simple + tests_random_sym

    run_tests(tests_all, "NumPy заглушка",
              solve_placeholder, is_iter=False)

    # все методы проверены

    run_tests(tests_all, "Метод Гаусса с выбором главного элемента",
              my_methods.gauss_with_pivoting, is_iter=False)
    run_tests(tests_all, "LU-разложение",
              my_methods.solve_lu, is_iter=False)
    run_tests(tests_all, "Метод Якоби",
              my_methods.solve_jacobi, is_iter=True, tol=1e-5)
    run_tests(tests_all, "Метод Зейделя",
              my_methods.seidel_method, is_iter=True)
    run_tests(tests_all_sym, "метод верхней релаксации",
              my_methods.sor_method, is_iter=True, omega=1)

    # не проходят для матриц, не являющихся положительно определенными
    run_tests(tests_all_sym, "метод градиентного спуска",
              my_methods.gradient_descent, is_iter=True)
    run_tests(tests_all_sym, "Метод минимальных невязок",
              my_methods.minimal_residual_method, is_iter=True)
    run_tests(tests_all_sym, "Метод сопряженных градиентов",
              my_methods.conjugate_gradient, is_iter=True)

    run_tests(tests_all, "Стабилизированный метод бисопряженных градиентов",
              my_methods.bicgstab, is_iter=True)

if __name__ == "__main__":
    main()
