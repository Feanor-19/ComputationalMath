import my_methods
import numpy as np
from matplotlib import pyplot as plt

def main():
    population = [
        76_212_168,
        92_228_496,
        106_021_537,
        123_202_624,
        132_164_569,
        151_325_798,
        179_323_175,
        203_211_926,
        226_545_805,
        248_709_873,
        281_421_906,
        308_745_538
    ]

    year_start = 1900
    year_final = 2010
    years = [x for x in range(year_start, year_final+1, 10)]

    target_year = 2020
    target_population = 331_449_281

    # Методы
    func_newton = my_methods.newton_interpolation(years, population)
    func_spline = my_methods.cubic_spline_interpolation(years, population)
    func_mnk    = my_methods.least_squares_poly(years, population, degree=11)

    # Погрешность в target
    forecast_newton = func_newton(target_year)
    forecast_spline = func_spline(target_year)
    forecast_mnk    = func_mnk(target_year)

    def rel_err(forecast):
        return abs(forecast - target_population)/target_population * 100

    rel_err_newton = rel_err(forecast_newton)
    rel_err_spline = rel_err(forecast_spline)
    rel_err_mnk    = rel_err(forecast_mnk)

    print(f"{rel_err_newton=}%, {rel_err_spline=}%, {rel_err_mnk=}%")

    # График
    x = np.linspace(year_start, target_year, 1000)

    y_newton = func_newton(x)
    y_spline = func_spline(x)
    y_mnk    = func_mnk(x)

    plt.errorbar(years, population, fmt="o")
    plt.errorbar(target_year, target_population, fmt="mo", label="Реальное значение")
    plt.plot(x, y_newton, label="Полином Ньютона")
    plt.plot(x, y_spline, label="Кубический сплайн")
    plt.plot(x, y_mnk, label="Метод наименьших квадратов")

    plt.minorticks_on()
    plt.grid()
    plt.title("Численность населения США в 1900-2020 гг")
    plt.xlabel("Год")
    plt.ylabel("Численность")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()

