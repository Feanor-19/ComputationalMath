import numpy as np
import random

# def generate_with_det(N):
#     """Генерирует матрицу с заданным определителем"""
#     A = np.random.uniform(-1, 1, (N, N))
#     current_det = np.linalg.det(A)
#     while abs(current_det) < 1e-2:  # Если определитель близок к 0
#         A += np.eye(N)  # Добавляем единичную матрицу
#         current_det = np.linalg.det(A)
#     return A

def generate_random_test_data(filepath, sizes, min_val=-10, max_val=10, seed=None, sym=False):
    """
    Генерирует случайные тестовые данные и записывает их в файл.

    Args:
        filepath: путь к файлу для записи
        sizes: список размеров N для каждого теста
        min_val: минимальное значение элементов матрицы
        max_val: максимальное значение элементов матрицы
        seed: seed для генератора случайных чисел (для воспроизводимости)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    with open(filepath, 'w') as f:
        for i, N in enumerate(sizes):
            # Генерируем невырожденную матрицу A
            # Для гарантии невырожденности используем матрицу с диагональным преобладанием
            A = np.random.uniform(min_val, max_val, (N, N))
            # A = generate_with_det(N)

            # симметризуем
            if (sym):
                for i in range(N-1):
                    for j in range(i+1, N):
                        A[i][j] = A[j][i]

            # Усиливаем диагональные элементы для обеспечения невырожденности
            for i in range(N):
                if (abs(A[i][i]) < 0.001):
                    A[i][i] += 1
                sign = A[i][i]/abs(A[i][i])
                for j in range(N):
                    if (j != i):
                        A[i][i] += sign*abs(A[i][j])


            # Генерируем случайный вектор x_true
            x_true = np.random.uniform(min_val, max_val, N)

            b = A @ x_true

            f.write(f"{N}\n")

            for row in A:
                f.write(" ".join(f"{val:.6f}" for val in row) + "\n")

            f.write(" ".join(f"{val:.6f}" for val in b) + "\n")
            f.write(" ".join(f"{val:.6f}" for val in x_true) + "\n")

            # Разделитель между тестами (кроме последнего)
            if i < len(sizes) - 1:
                f.write("---\n")

    print(f"Сгенерировано {len(sizes)} тестов в файл {filepath}")

def main():
    generate_random_test_data("lab2\\checker_tests\\rand.txt",
                              [6, 5, 5, 4, 4, 4, 3, 3, 3, 3])
    generate_random_test_data("lab2\\checker_tests\\rand_sym.txt",
                              [6, 5, 5, 4, 4, 4, 3, 3, 3, 3], sym=True)

if __name__ == "__main__":
    main()
