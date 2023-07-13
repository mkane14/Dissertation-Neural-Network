import os
import numpy as np
import math
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from scipy.stats import norm
from scipy.optimize import curve_fit


#Function to return the mean
def mean(arr):
    return sum(arr) / len(arr)


#Function to return the standard deviation of a dataset
def std_dev(arr):
    arr_mean = mean(arr)
    return (sum([((i - arr_mean)**2) for i in arr]) / (len(arr) - 1)) ** 0.5


#Standard Gaussian function
def gaussian(x, mu, sigma, A):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


#This function generates the weights and bias's for the network by drawing them at random from some normal distribution
def initialization(input_size_, hidden_size_, Mu_, STD_, nl, weightorbias):

    if weightorbias == "weight":
        Wname = 'W{}'.format(nl)
        out = tf.Variable(tf.random.normal([input_size_, hidden_size_], mean=Mu_, stddev=STD_), name=Wname)

    if weightorbias == "bias":
        bname = 'b{}'.format(nl)
        out = tf.Variable(tf.random.normal([hidden_size_], mean=Mu_, stddev=STD_), name=bname)

    return out


#This function propagates the signal through the network
def forward(X, i, activation_function, depth_, Ws, Bs, Layer_output_matrix):

    with tf.GradientTape() as tape:
        tape.watch([param for param in [Ws[i, j] for j in range(depth_)] + [Bs[i, j] for j in range(depth_)]])

        #Setting the preactivation of the first hidden layer as the input data set
        A = X

        for j in range(depth_):

            #Calculating the preactivation and activation function for a given hidden layer
            Z = tf.add(tf.matmul(A, Ws[i, j]), Bs[i, j])
            A = activation_function(Z)

            #This is taking the output distribution of each layer
            Layer_output_matrix[i, j] = Z

        #Output layer distribution function
        Z_last = Z

    gradients = tape.gradient(Z_last, [Ws[i, j] for j in range(depth_)] + [Bs[i, j] for j in range(depth_)])

    return Z_last, gradients, Layer_output_matrix


#Used to exract data from the weights and bias generation, allows for histogram plot
def bootobj_extr(layer_, el1_, el2_, Nboot, Ws, Bs):

    A = np.zeros(Nboot)
    B = np.zeros(Nboot)

    for i in range(Nboot):
        matrW = Ws[i, layer_]
        A[i] = matrW[el1_, el2_]
        vecB = Bs[i, layer_]
        B[i] = vecB[el1_]

    return A, B


#This function initiates the neural network with the given inputs and gives the statistics of each step
def initialise_network(activation_function, width, depth, Cb, Cw, Mu, Nboot):

    #Input and output layer size (number of neurons)
    input_size = 1
    output_size = 1

    #Prints the networks input statistics
    net_stat = f"Nboot={Nboot}, Cb={Cb} & Cw={Cw}, Mu={round(Mu, 2)}"
    print(f"Input Gaussian Values: {net_stat}")

    #These if statements are used in classifying the type of plot so they're saved to the correct place, nothing more
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

    lsizes = np.concatenate(([input_size], np.repeat(width, depth - 1), [output_size]))

    #Prints the network architecture
    net_arch = f"Width = {width} & Depth = {depth}"
    print(f"Network Architecture: {net_arch} \n{lsizes}")

    #Generating the weights and bias matrices
    Ws = np.zeros((Nboot, depth), dtype=object)
    Bs = np.zeros((Nboot, depth), dtype=object)

    #Generating the weights and biases for each layer for all instantiations of the network
    #Nboot is the number of times the network is initiated
    for i in range(Nboot):
        for l in range(depth):
            Bs[i, l] = initialization(lsizes[l], lsizes[l + 1], Mu, np.sqrt(Cb) / lsizes[l], l, weightorbias="bias")
            Ws[i, l] = initialization(lsizes[l], lsizes[l + 1], Mu, np.sqrt(Cw) / lsizes[l], l, weightorbias="weight")

    #Filtering the input bias and weight distribution data to plot on a histogram
    Whist, Bhist = bootobj_extr(1, 1, 1, Nboot=Nboot, Ws=Ws, Bs=Bs)

    #Plotting the bias input distribution
    bias_input_dist(Bhist, width, depth, Cw, Cb, Mu, Nboot, plot_type)

    #Plotting the weight input distribution
    weight_input_dist(Whist, width, depth, Cw, Cb, Mu, Nboot, plot_type)

    # Input quantities
    X = tf.constant([[30]], dtype=tf.float32)

    # We compute the output for each event for an input
    Lom = np.zeros((Nboot, depth), dtype=object)       #L'th layer output matrix
    output_array = []
    grads_array = np.array([])

    #Propagates the signal throughout the network for every instantiation
    for i in range(Nboot):
        Y, Grad, Layer_out = forward(X, i, activation_function, depth, Ws=Ws, Bs=Bs, Layer_output_matrix=Lom)
        output_array.append(Y.numpy()[0][0])

        grads_np = np.concatenate([g.numpy().flatten() for g in Grad])
        grads_array = np.vstack([grads_array, grads_np]) if grads_array.size else grads_np

    #Plots the final layer output distribution
    final_out_dist(output_array, width, depth, Cw, Cb, Mu, Nboot, plot_type)

    #Generates the output distribution per layer of the network
    for i in range(depth):
        A = np.zeros(Nboot)
        for j in range(Nboot):
            matrLom = Lom[j, i]
            A[j] = matrLom[0, 0]

        #Plots the output distribution for every layer
        output_dist_per_layer(A, width, i, Cw, Cb, Mu, Nboot, plot_type)

    return


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

    os.makedirs(f"Plots\Varied_Architecture_Plots\{plot_type}_Plots\Bias_In_Dist_Plots\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}", exist_ok=True)
    plt.savefig(f'Plots\Varied_Architecture_Plots\{plot_type}_Plots\Bias_In_Dist_Plots\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}\{width}w {depth}d.png')
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

    os.makedirs(f"Plots\Varied_Architecture_Plots\{plot_type}_Plots\Weight_In_Dist_Plots\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}", exist_ok=True)
    plt.savefig(f"Plots\Varied_Architecture_Plots\{plot_type}_Plots\Weight_In_Dist_Plots\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}\{width}w {depth}d.png")
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
    plt.text(xlim, 0.82*ylim, f"Output Stats:\nMean={round(coeff[0], 4)}\nFitted STD={round(STD_fit, 4)}\nData STD={round(STD_array, 4)}\nAmp={round(coeff[2], 4)}", fontsize=9)

    #Saves and files plot to a new or existing directory
    os.makedirs(f"Plots\Varied_Architecture_Plots\{plot_type}_Plots\Output_Dist_Plots\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}", exist_ok=True)
    plt.savefig(f"Plots\Varied_Architecture_Plots\{plot_type}_Plots\Output_Dist_Plots\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}\{width}w {depth}d.png")
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
            print(f"Layer {i+1} output distribution is not Gaussian")
        if p >= 0.05:
            title_add = "Gaussian"
            print(f"Layer {i+1} output distribution is Gaussian")

        #Take parameters of our histogram80
        counts, bins, _ = plt.hist(A, bins=100)

        ylim = max(counts)
        xlim = min(bins)
        xlim2 = max(bins)

        # Plot the histogram and the fitted Gaussian function
        x = np.linspace(xlim, xlim2, 1000)

        #Statistics of output distribution
        p0 = [(xlim + xlim2) / 2, xlim2 - xlim, 1]
        coeff, var_matrix = curve_fit(gaussian, bins[:-1], counts, p0=p0)
        STD_fit = np.abs(coeff[1])
        STD_array = std_dev(A)

        #Plotting and labelling the data
        plt.plot(x, gaussian(x, coeff[0], STD_fit, coeff[2]), 'r--', label='fit')
        plt.hist(A, bins=100, color="blue")
        plt.xlabel(f'Layer {i+1} Output')
        plt.ylabel('Frequency')
        plt.suptitle(f'Layer {i+1} Output Distribution ({title_add})')
        plt.title(f'p = {round(p, 6)}')
        plt.text(0.55 * xlim2, 0.82 * ylim, f"Input Params:\n{plot_type}\n{width}w @ Layer {i+1}\nCw={Cw} & Cb={Cb}\nNboot={Nboot}", fontsize=9)
        plt.text(xlim, 0.82 * ylim, f"Output Stats:\nMean={round(coeff[0], 4)}\nFitted STD={round(STD_fit, 4)}\nData STD={round(STD_array, 4)}\nAmp={round(coeff[2], 4)}", fontsize=9)

        #Saves and files plot to a new or existing directory
        os.makedirs(f"Plots\Output_Per_Layer_Plots\{plot_type}_Plots\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}", exist_ok=True)
        plt.savefig(f"Plots\Output_Per_Layer_Plots\{plot_type}_Plots\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}\Layer {i+1}.png")
        #plt.show()
        plt.clf()