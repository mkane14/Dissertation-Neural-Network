import numpy as np
import os
import matplotlib.pyplot as plt
import plot
from scipy import integrate
from scipy.optimize import curve_fit
import math


def avg_rho_sq(activation_function, k):
    coeff = 1 / np.sqrt(2 * math.pi * k)
    integrand = lambda z: np.exp((-1 / 2) * (1 / k) * (z ** 2)) * (activation_function(z) ** 2)
    integral = integrate.quad(integrand, -np.inf, np.inf)
    value = coeff * integral[0]
    est_error = integral[1]

    return value, est_error


def recursion(activation_function, input_x, depth, Cw, Cb):

    plot_type = plot.determine_plot_type(activation_function)

    kl_array = []
    kl_err_array = []
    l_array = []
    kl = Cb + Cw * (input_x ** 2)
    kl_array.append(kl)
    l_array.append(1)

    for l in range(2, depth+1):
        l_array.append(l)
        avg_rho, error = avg_rho_sq(activation_function, kl)
        kl = Cb + (Cw * (avg_rho))
        kl_array.append(kl)
        kl_err_array.append(error)

    #print(f"Potential K Fixed Point: k({l}) = {kl} for input = {input_x}")

    plt.clf()
    label = str("$K^{(l)}$") + str(f" for x = {input_x}")
    plt.plot(l_array, kl_array, color="k", label=label)
    plt.legend()
    plt.xlabel("Layer [l]")
    plt.ylabel("$K^{(l)}$")
    #plt.title("Kernel as a function of Layer (n -> inf)")
    os.makedirs(f"Analytical Plots\{plot_type}\Recursion\Cw={round(Cw, 2)} & Cb={Cb}\ x={input_x}", exist_ok=True)
    plt.savefig(f"Analytical Plots\{plot_type}\Recursion\Cw={round(Cw, 2)} & Cb={Cb}\ x={input_x}\Depth {depth}.png", bbox_inches="tight")

    return kl_array