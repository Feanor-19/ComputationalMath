import math
import matplotlib.pyplot as plt
import numpy as np
import methods

# функция для рисования графика f(x) на указанном интервале [a, b]
def plot_onedim(f, a, b, numpoints=1000):
    _, ax = plt.subplots()
    x = np.linspace(a, b, numpoints)
    y = f(x)
    ax.plot(x, y, 'b-')

    ax.axline((0,0),slope=0,c='black')
    ax.axvline(x=0, c='black')

    axes_max = int(max(abs(max(x)), abs(max(y)), abs(min(x)), abs(min(y))))
    axes_min = -axes_max

    plt.xticks([i for i in range(axes_min-1, axes_max+1)])
    plt.yticks([i for i in range(axes_min-1, axes_max+1)])

    ax.grid()
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

# функция для рисования двумерных кривых на плоскости в указанном прямоугольнике
def plot_2d_curve(F, x_range=(-5, 5), y_range=(-5, 5), resolution=1000):
    # Если передана одна функция, преобразуем в список
    # functions = [functions]

    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[1], y_range[0], resolution)  # обратный порядок для корректной ориентации
    X, Y = np.meshgrid(x, y)

    plt.contour(X, Y, F([X,Y])[0], levels=[0], colors='orange')
    plt.contour(X, Y, F([X,Y])[1], levels=[0], colors='blue')

    plt.axline((0,0),slope=0,c='black')
    plt.axvline(x=0, c='black')

    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.show()

TOL = 1e-7
MAX_ITER = 1000

line_types = ['-', '--', ':']
colors = ['blue', 'orange', 'green', 'red', 'magenta']
def get_line_type(i):
    return line_types[i % len(line_types)]

def get_color(i):
    return colors[i % len(colors)]

def solve_onedim_all(f, segments):
    methods1d_list = [
        (methods.method_bisection, "Метод бисекции"),
        (methods.method_chord, "Метод хорд")
    ]

    methods2d_list = [
        (methods.method_simple_iteration, "Метод простой итерации"),
        (methods.method_newton, "Метод Ньютона"),
        (methods.method_modified_newton, "Модифицированный метод Ньютона")
    ]


    to_plot_resds = []
    for i in range(len(segments)):
        a = segments[i][0]
        b = segments[i][1]

        # for m in methods1d_list:
        for j in range(len(methods1d_list)):
            m = methods1d_list[j]
            try:
                print("\t"+m[1] + f", отрезок [{a}, {b}]:")
                x, resds = m[0](f, a, b, tol=TOL, max_iter=MAX_ITER)
                print(x)
                to_plot_resds.append((resds, x, m[1], get_line_type(i), get_color(j)))
            except Exception as err:
                print(f"Exception {err=}, {type(err)=}")
        # for m in methods2d_list:
        for j in range(len(methods2d_list)):
            m = methods2d_list[j]
            try:
                print("\t"+m[1] + f", начальное приближение {(a+b)/2}:")
                x, resds = m[0](f, [(a+b)/2], tol=TOL, max_iter=MAX_ITER)
                print(x)
                to_plot_resds.append((resds, x[0], m[1], get_line_type(i), get_color(len(methods1d_list)+j)))
            except Exception as err:
                print(f"Exception {err=}, {type(err)=}")


    for i in range(len(to_plot_resds)):
        elem = to_plot_resds[i]
        plt.semilogy(elem[0], elem[3], color=elem[4], linewidth=2, label=(elem[2]+": x="+str(elem[1])))
    plt.legend()
    plt.xlabel('Итерация')
    plt.ylabel('Норма невязки')
    plt.xticks([i for i in range(0, 1+max( [ len(a[0]) for a in to_plot_resds ] ))])
    plt.grid(True)
    plt.show()


def process_onedim():
    # выбранные нелинейные уравнения
    f_1 = lambda x: np.array(np.power(x,2) - np.exp(x)/5)
    f_2 = lambda x: np.array(x*np.power(2,x)-1)

    # plot_onedim(f_1, -4, 5) # -> roots in [-1, 0], [0, 1], [4, 5]
    # plot_onedim(f_2, -7, 2) # -> root in [0, 1]

    print("\t\tНелинейное уравнение 1:\n")
    solve_onedim_all(f_1, [(-1, 0), (0, 1), (4, 5)])
    print(f"\n---\n")
    print("\t\tНелинейное уравнение 2:")
    solve_onedim_all(f_2, [(0, 1)])

def solve_twodim_all(F, init_points):
    methods_list = [
        (methods.method_simple_iteration, "Метод простой итерации"),
        (methods.method_newton, "Метод Ньютона"),
        (methods.method_modified_newton, "Модифицированный метод Ньютона")
    ]

    to_plot_resds = []

    for i in range(len(init_points)):
        init_point = init_points[i]
        for j in range(len(methods_list)):
            m = methods_list[j]
            try:
                print(m[1] + f", начальное приближение: " + str(init_point))
                x, resds = m[0](F, np.array(init_point), tol=TOL, max_iter=MAX_ITER)
                print(x)
                to_plot_resds.append((resds, x, m[1], get_line_type(i), get_color(j)))
            except Exception as err:
                print(f"Exception {err=}, {type(err)=}")

    for elem in to_plot_resds:
        plt.semilogy(elem[0], elem[3], color=elem[4], linewidth=2, label=(elem[2]+": (x, y)="+str(elem[1])))
    plt.legend()
    plt.xlabel('Итерация')
    plt.ylabel('Норма невязки')
    xticks_max = 1+max( [ len(a[0]) for a in to_plot_resds ] )
    xticks_step = int(xticks_max / 10 if xticks_max > 50 else 1)
    plt.xticks([i for i in range(0, xticks_max, xticks_step)])
    plt.grid(True)
    plt.show()

def process_twodim():
    # выбранные нелинейные системы уравнений
    # F_1(x, y) = (x-cos(y)-3, cos(x-1)+y-0.5)
    F_1 = lambda X: np.array([X[0]-np.cos(X[1])-3, np.cos(X[0]-1)+X[1]-0.5])
    # F_2(x, y) = ((x-1.4)**2 - (y-0.6)**2 - 1, 4.2*(x**2) + 8.8*(y**2) - 1.42)
    F_2 = lambda X: np.array([(X[0]-1.4)**2 - (X[1]-0.6)**2 - 1, 4.2*(X[0]**2) + 8.8*(X[1]**2) - 1.42])

    # plot_2d_curve(F_1, x_range=(-3, 9), y_range=(-5, 5)) # -> (3.5, 1.5)
    # plot_2d_curve(F_2, x_range=(-2, 4), y_range=(-2, 2)) # -> (0, -0.4), (0.4, 0.4)

    print("Нелинейная система уравнений 1:")
    solve_twodim_all(F_1, [[3.5, 1.5]])
    print(f"\n---\n")
    print("Нелинейная система уравнений 2:")
    solve_twodim_all(F_2, [[-0.1, -0.4], [0.4, 0.4]])

def main():
    # process_onedim()
    process_twodim()

if __name__ == "__main__":
    main()
