#This code plots the network inputs and outputs and displays their statistics
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from scipy.optimize import curve_fit
from scipy.stats import normaltest

#Function to return the mean
def mean(arr):
    return sum(arr) / len(arr)


#Calculates the standard deviation for a given input array of data
def std_dev(arr):
    arr_mean = mean(arr)
    return (sum([((i - arr_mean)**2) for i in arr]) / (len(arr) - 1)) ** 0.5


#Standard Gaussian function
def gaussian(x, mu, sigma, A):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


#Purely for file save locations and
def determine_plot_type(activation_function):

    #These "if" statements are used in classifying the type of plot
    if activation_function == tf.nn.relu:
        plot_type = "RELU"
    if activation_function == tf.nn.tanh:
        plot_type = "Tanh"
    if activation_function == tf.nn.selu:
        plot_type = "SELU"
    if activation_function == tf.nn.gelu:
        plot_type = "GELU"
    if activation_function == tf.nn.sigmoid:
        plot_type = "Sigmoid"
    if activation_function == tf.nn.elu:
        plot_type = "ELU"

    return plot_type


#Plots the normal distribution of the input bias's
def bias_input_dist(Bhist, width, depth, Cw, Cb, Mu, Nboot, plot_type):
    # Test of the gaussianity of bias
    stat, p = normaltest(Bhist)
    # Print the result of the test
    if p < 0.05:
        title_add = "Not Gaussian"
        print("->Input bias distribution is not Gaussian.")
    else:
        title_add = "Gaussian"
        print("->Input bias distribution is Gaussian.")

    # Print the input distributions of the weights and bias
    plt.hist(Bhist, bins=100)
    plt.xlabel('Bias')
    plt.ylabel('Frequency')
    plt.suptitle(f'Bias Input Distribution ({title_add})')
    bias_plot_name = f"{width}w & {depth}d, Nboot = {Nboot}, Cb={round(Cb, 2)}, Mu = {round(Mu, 2)}"
    plt.title(f"{bias_plot_name}")

    os.makedirs(f"Plots\Input_Dists\{plot_type}\Bias_In_Dist\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}", exist_ok=True)
    plt.savefig(f'Plots\Input_Dists\{plot_type}\Bias_In_Dist\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}\{width}w {depth}d.png')
    # plt.show()
    plt.clf()


#Plots the input distribution for the randomly generated weights
def weight_input_dist(Whist, width, depth, Cw, Cb, Mu, Nboot, plot_type):
    # Test the gaussianity of the weights input distribution
    stat, p = normaltest(Whist)
    if p < 0.05:
        title_add = "Not Gaussian"
        print("->Input weight distribution is not Gaussian")
    else:
        title_add = "Gaussian"
        print("->Input weight sample distribution is Gaussian")

    plt.hist(Whist, bins=100)
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.suptitle(f'Weight Input Distribution ({title_add})')
    weight_plot_name = f"{width}w & {depth}d, Nboot = {Nboot}, Cw={round(Cw, 2)}, Mu = {round(Mu, 2)}"
    plt.title(f"{weight_plot_name}")

    os.makedirs(f"Plots\Input_Dists\{plot_type}\Weight_In_Dist\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}", exist_ok=True)
    plt.savefig(f"Plots\Input_Dists\{plot_type}\Weight_In_Dist\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}\{width}w {depth}d.png")
    # plt.show()
    plt.clf()


#Plots the final layer output distribution
def final_out_dist(output_array, width, depth, Cw, Cb, Mu, Nboot, plot_type):

    #Test for gaussianity
    stat, p = normaltest(output_array)

    #Print the result of the test
    if p < 0.05:
        title_add = "Not Gaussian"
        print("->Final layer output distribution is not Gaussian")
    if p >= 0.05:
        title_add = "Gaussian"
        print("->Final layer output distribution is Gaussian")

    #Take parameters of our histogram
    counts, bins, _ = plt.hist(output_array, bins=100)

    ylim = max(counts)
    xlim = min(bins)
    xlim2 = max(bins)

    #Statistics of the output distribution
    x = np.linspace(xlim, xlim2, 1000)
    p0 = [(xlim + xlim2) / 2, xlim2 - xlim, 1]
    coeff, var_matrix = curve_fit(gaussian, bins[:-1], counts, p0=p0)
    STD_fit = np.abs(coeff[1])
    STD_array = std_dev(output_array)

    # Plot the histogram and the fitted Gaussian function
    plt.plot(x, gaussian(x, coeff[0], STD_fit, coeff[2]), 'r--', label='fit')
    plt.hist(output_array, bins=100, color='blue')
    plt.xlabel('Output')
    plt.ylabel('Frequency')
    plt.suptitle(f'Output Distribution ({title_add})')
    plt.title(f'p = {round(p, 6)}')

    plt.text(0.55*xlim2, 0.82*ylim, f"Input Params:\n{plot_type}\n{width}w & {depth}d\nCw={Cw} & Cb={Cb}\nNboot={Nboot}", fontsize=9)
    plt.text(xlim, 0.82*ylim, f"Fit Output Stats:\nMean={round(coeff[0], 4)}\nFitted STD={round(STD_fit, 4)}\nData STD={round(STD_array, 4)}\nAmp={round(coeff[2], 4)}", fontsize=9)

    #Saves and files plot to a new or existing directory
    os.makedirs(f"Plots\Output_Dists\Final_Layer\{plot_type}\{width}w {depth}d", exist_ok=True)
    plt.savefig(f"Plots\Output_Dists\Final_Layer\{plot_type}\{width}w {depth}d\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}.png")
    #plt.show()
    plt.clf()

    #Print gaussian plot parameters
    #print("Mean =", coeff[0])
    #print("Standard dev. from data=", STD_array)
    #print("Standard dev. from fit=", STD_fit)
    #print("Amplitude =", coeff[2])
    #print("Output Dist. Variance =", (coeff[1]**2))


#Funcion to plot the output distribution of each layer in a given network
def output_dist_per_layer(A, width, i, Cw, Cb, Mu, Nboot, plot_type):

    #Plotting the output distributions for each layer output within the neural network
    stat, p = normaltest(A)

    if p < 0.05:
            title_add = "Not Gaussian"
            #print(f"Layer {i+1} output distribution is not Gaussian")
    if p >= 0.05:
            title_add = "Gaussian"
            #print(f"Layer {i+1} output distribution is Gaussian")

    #Take parameters of our histogram80
    counts, bins, _ = plt.hist(A, bins=100)

    ylim2 = max(counts)
    xlim = min(bins)
    xlim2 = max(bins)

    #Statistics of output distribution
    p0 = [(xlim + xlim2) / 2, xlim2 - xlim, 1]
    coeff, var_matrix = curve_fit(gaussian, bins[:-1], counts, p0=p0)
    STD_fit = np.abs(coeff[1])
    STD_array = std_dev(A)

    #Find the full width half maximum
    Mean = coeff[0]
    sigma = STD_fit
    sqrt_vals = (sigma*np.sqrt(2*(np.log(2))))
    x_pos = Mean + sqrt_vals
    x_neg = Mean - sqrt_vals
    FWHM = np.abs(x_pos - x_neg)
    #print(f"FWHM: {FWHM}")

    #Histogram spacings
    x = np.linspace(xlim, xlim2, 1000)
    ylim_fit = coeff[2]           #Fitted Gaussian amplitude
    y = np.linspace(0, ylim_fit, 100)
    y_space = np.linspace(ylim_fit/2, ylim_fit/2, 1000)
    x_pos_space = np.linspace(x_pos, x_pos, 100)
    x_neg_space = np.linspace(x_neg, x_neg, 100)

    #Plotting and labelling the data
    plt.plot(x, gaussian(x, coeff[0], STD_fit, coeff[2]), 'r--', label='fit')
    plt.plot(x_pos_space, y, "k-")
    plt.plot(x_neg_space, y, "k-")
    plt.plot(x, y_space, "k-")
    plt.hist(A, bins=100, color="blue")
    plt.xlabel(f'Layer {i+1} Output')
    plt.ylabel('Frequency')
    plt.suptitle(f'Layer {i+1} Output Distribution ({title_add})')
    plt.title(f'p = {round(p, 6)}')
    plt.text(0.55 * xlim2, 0.82 * ylim2, f"Input Params:\n{plot_type}\n{width}w @ Layer {i+1}\nCw={Cw} & Cb={Cb}\nNboot={Nboot}", fontsize=9)
    plt.text(xlim, 0.78 * ylim2, f"Fit Output Stats:\nMean={round(coeff[0], 4)}\nFitted STD={round(STD_fit, 4)}\nData STD={round(STD_array, 4)}\nAmp={round(coeff[2], 4)}\nFWHM={round(FWHM, 4)}", fontsize=9)

    #Saves and files plot to a new or existing directory
    os.makedirs(f"Plots\Output_Dists\Each_Layer\{plot_type}\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}", exist_ok=True)
    plt.savefig(f"Plots\Output_Dists\Each_Layer\{plot_type}\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}\Layer {i+1}.png")
    #plt.show()
    plt.clf()

    return FWHM


#Plots the Full Width Half Max of the output distribution as a function of depth
def gaussian_width_and_depth(width, depth, Cw, Cb, Mu, Nboot, FWHM_array, plot_type):

    x_axis = range(1, depth + 1)
    plt.plot(x_axis, FWHM_array, )
    plt.xlabel('Layer')
    plt.ylabel('Gaussian FWHM')
    plt.title("FWHM vs Layer")
    plt.text(0.75 * depth, 0.75 * FWHM_array[0], f"Input Params:\n{plot_type}\n{width}w & {depth}d\nCw={Cw} & Cb={Cb}\nNboot={Nboot}", fontsize=10)

    os.makedirs(f"Plots\FWHM_vs_Depth\{plot_type}\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}", exist_ok=True)
    plt.savefig(f"Plots\FWHM_vs_Depth\{plot_type}\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}\{width}w {depth}d")
    plt.clf()

    return