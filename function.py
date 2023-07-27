#This code contains everything to do with the neural network and extracting its statistics
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import plot
from scipy.optimize import curve_fit
from scipy import integrate
import os

#Generates the weights and bias's for the network by drawing them at random from some normal distribution
def initialization(input_size_, hidden_size_, Mu_, STD_, nl, weightorbias):

    if weightorbias == "weight":
        Wname = 'W{}'.format(nl)
        out = tf.Variable(tf.random.normal([input_size_, hidden_size_], mean=Mu_, stddev=STD_), name=Wname)

    if weightorbias == "bias":
        bname = 'b{}'.format(nl)
        out = tf.Variable(tf.random.normal([hidden_size_], mean=Mu_, stddev=STD_), name=bname)

    return out


#Linear activation function
def linear(Z):
    A = Z
    return A


#This function propagates the signal through the network
def forward(X, i, activation_function, depth_, Ws, Bs, Layer_out_matrix):

    with tf.GradientTape() as tape:
        tape.watch([param for param in [Ws[i, j] for j in range(depth_)] + [Bs[i, j] for j in range(depth_)]])

        #Setting the preactivation of the first hidden layer as the input data set
        A = X

        for l in range(depth_):
            #Calculating the preactivation and activation function for a given hidden layer
            Z = tf.add(tf.matmul(A, Ws[i, l]), Bs[i, l])
            A = activation_function(Z)

            #Appending the output distribution for each layer of the network
            Layer_out_matrix[i, l] = Z

        Z_last = Z

    return Z_last


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


def avg_zl_sq(g):

    coeff = 1 / (np.sqrt(2 * math.pi * g))
    integrand = lambda z: (np.exp((-1 / 2) * (1 / g) * (z ** 2))) * (z ** 2)
    integral = integrate.quad(integrand, -np.inf, np.inf)
    value = coeff * integral[0]
    est_error = integral[1]

    return value, est_error


def zl_sq(inv_w, kl, g1l, correc):
    return kl + (inv_w * g1l) + (correc*(inv_w**2))


#This function initiates the neural network with the given inputs and gives the statistics of each step
def initialise_network(activation_function, width, depth, Cw, Cb, Mu, Nboot, rescale):

    print("\n----Initiating----")

    #Input and output layer size (number of neurons)
    input_size = 1
    output_size = 1

    plot_type = plot.determine_plot_type(activation_function)

    #Prints the networks input statistics
    net_stat = f"Nboot={Nboot}, Cb={Cb} & Cw={Cw}, Mu={round(Mu, 2)}"
    print(f"Input Gaussian Values: {net_stat}")

    #Creates an array with sizes of each layer as each value
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
            if rescale == True:
                Bs[i, l] = initialization(lsizes[l], lsizes[l + 1], Mu, np.sqrt(Cb) / lsizes[l], l, weightorbias="bias")
                Ws[i, l] = initialization(lsizes[l], lsizes[l + 1], Mu, np.sqrt(Cw) / lsizes[l], l, weightorbias="weight")
            if rescale == False:
                Bs[i, l] = initialization(lsizes[l], lsizes[l + 1], Mu, np.sqrt(Cb), l, weightorbias="bias")
                Ws[i, l] = initialization(lsizes[l], lsizes[l + 1], Mu, np.sqrt(Cw), l, weightorbias="weight")

    #Filtering the input bias and weight distribution data to plot on a histogram
    #Whist, Bhist = bootobj_extr(1, 1, 1, Nboot=Nboot, Ws=Ws, Bs=Bs)

    #Plotting the bias input distribution
    #plot.bias_input_dist(Bhist, width, depth, Cw, Cb, Mu, Nboot, plot_type)

    #Plotting the weight input distribution
    #plot.weight_input_dist(Whist, width, depth, Cw, Cb, Mu, Nboot, plot_type)

    # Input quantities
    X = tf.constant([[30]], dtype=tf.float32)

    #L'th layer output matrix
    Lom = np.zeros((Nboot, depth), dtype=object)

    #Final layer output layer array
    output_array = []

    #Propagates the signal throughout the network for every instantiation
    for i in range(Nboot):
        Z = forward(X, i, activation_function, depth, Ws=Ws, Bs=Bs, Layer_out_matrix=Lom)
        output_array.append(Z.numpy()[0][0])

    #Plot the final layer output distribution
    #plot.final_out_dist(output_array, width, depth, Cw, Cb, Mu, Nboot, plot_type, rescale)

    #Arrays to store values for FWHM of fitted data
    FWHM_array = []
    STD_fit_array = []
    STD_data_array = []
    data_zsq = []

    #Filtering the array data for the output distribution of each layer so we can plot it
    for l in range(depth):
        Z = np.zeros(Nboot)
        zsq = 0
        for i in range(Nboot):
            matrLom = Lom[i, l]
            Z[i] = matrLom[0, 0]
            # Calculating <Zsq> directly from the output data for every layer
            zsq += (Z[i]**2)/Nboot
        data_zsq.append(zsq)

        #Plots the output distribution for every layer and appending each layers FWHM
        FWHM, STD_fit, STD_data = plot.output_dist_per_layer(Z, width, depth, l, Cw, Cb, Mu, Nboot, plot_type, rescale)
        FWHM_array.append(FWHM)
        STD_fit_array.append(STD_fit)
        STD_data_array.append(STD_data)

    #Plotting the FWHM vs depth of the network
    #plot.gaussian_width_and_depth(width, depth, Cw, Cb, Mu, Nboot, FWHM_array, plot_type)

    ### <Zsq> using the fitted output data and integral ###
    fit_zsq = []
    error_fit_zsq = []

    # Calculating the <Zsq> directly from the variance of the fitted gaussian output distributions for every layer
    for sigma in STD_fit_array:
        g = sigma**2
        outputs = avg_zl_sq(g)
        fit_zsq.append(outputs[0])
        error_fit_zsq.append(outputs[1])

    print("----Complete----")

    return fit_zsq, error_fit_zsq, data_zsq

# Function to determine kl (g0l), g1l and the order(1/nsq) correction to eq's 5.108-5.110 for finite width networks for a single input
def numerical_analysis(activation_function, initial_width, final_width, depth, Cw, Cb, Mu, Nboot, rescale):

    # Determines plot from activation function - just for plotting and saving
    plot_type = plot.determine_plot_type(activation_function)

    # The difference in the initial and final width used to define the dimensions of the arrays
    width_diff = final_width - initial_width + 1

    # Defining the matrices used to store values for different parameters
    fit_zsq_matrix = np.zeros((depth, width_diff), dtype=object)
    err_fit_zsq_matrix = np.zeros((depth, width_diff), dtype=object)
    data_zsq_matrix = np.zeros((depth, width_diff), dtype=object)

    # "Data" is using the raw output through signal propagation from layer to layer - actual network output
    data_kl_array = []
    data_g1l_array = []
    data_correc_array = []

    # "Fit" data uses the fitted gaussian curves on the output distribution functions for each layer - best fitted network outputs
    fit_kl_array = []
    fit_g1l_array = []
    fit_correc_array = []

    # Initiating the network for the variety of widths defined in order to allow for plots
    for w, j in enumerate(range(initial_width, final_width+1)):

        # Storing the network outputs for every width
        zsq_fit, err_fit_zsq, zsq_data = initialise_network(activation_function, j, depth, Cw, Cb, Mu, Nboot, rescale)

        # Storing the network outputs for every layer
        for l in range(depth):
            fit_zsq_matrix[l, w] = zsq_fit[l]
            err_fit_zsq_matrix[l, w] = err_fit_zsq[l]
            data_zsq_matrix[l, w] = zsq_data[l]

    # Starting the numerical analysis
    print("----Numerical Analysis----")

    for l in range(depth):

        #Arrays to store values
        l_fit_matrix = []
        error_l_fit_matrix = []
        l_data_matrix = []
        inv_width_axis = []

        #Picking the specific values required from the matrices that allow for us to plot <Zsq> against 1/w
        for w, j in enumerate(range(initial_width, final_width+1)):
            l_fit_matrix.append(fit_zsq_matrix[l, w])
            error_l_fit_matrix.append(err_fit_zsq_matrix[l, w])
            l_data_matrix.append(data_zsq_matrix[l, w])
            inv_width_axis.append(1/j)

        print(f"\n----Raw network Out. Dist. analysis for Layer {l+1}----")

        # Best fitting the stored values in order to match "zl_sq", this gives us the best values for kl, g1l and the correction using the raw network output
        plt.clf()
        data_params, _ = curve_fit(zl_sq, inv_width_axis, l_data_matrix)
        data_kl = data_params[0]
        data_kl_array.append(data_kl)
        data_g1l = data_params[1]
        data_g1l_array.append(data_g1l)
        data_correc = data_params[2]
        data_correc_array.append(data_correc)

        # Printing these values for the specific layers
        print(f"k{l + 1}: ", data_kl, f"\ng[1]({l + 1}): ", data_g1l, f"\n1/nsq Correction Coeff: ", data_correc)

        # Now plotting these stored outputs as a scatter graph
        plt.clf()
        plt.scatter(inv_width_axis, l_data_matrix, marker="x", label="Data Points")

        # Using the best fit parameters to make a best fit curve plot, allows for visualisation of how well the kl, g1l and corrections fit the data
        x_axis = np.linspace(1 / initial_width, 1 / final_width, 500)
        data_curve_y_array = []

        for i in x_axis:
            data_curve_y = zl_sq(i, data_kl, data_g1l, data_correc)
            data_curve_y_array.append(data_curve_y)

        plt.plot(x_axis, data_curve_y_array, "r--", label="Best Fit")

        # Properly labelling and saving the assosciated plot
        plt.legend()
        plt.xlabel(f'Inverse Width')
        plt.ylabel('<Zsq>')
        plt.suptitle(f'<Zsq> vs 1/w (Raw Network OutDist Data) (Layer {l + 1})')
        plt.title(f'k({l + 1}) = {round(data_kl, 2)}, g[1]({l + 1}) = {round(data_g1l, 2)}, Correction Scale = {round(data_correc, 2)}')
        os.makedirs(f'Plots\Analysis_Plots\{plot_type}\Avg_Zsq\Raw_OutDist_Data_Plots\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}\{initial_width}w-{final_width}w & {depth}d', exist_ok=True)
        plt.savefig(f'Plots\Analysis_Plots\{plot_type}\Avg_Zsq\Raw_OutDist_Data_Plots\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}\{initial_width}w-{final_width}w & {depth}d\Layer {l + 1}.png')

        # Starting the numerical analysis for the raw network output data
        print(f"\n----Fitted Gaussian Out. Dist. data analysis for Layer {l+1}----")

        # Best fitted kl, g1l and corrections to the networks fitted output
        plt.clf()
        fit_params, _ = curve_fit(zl_sq, inv_width_axis, l_fit_matrix, sigma = error_l_fit_matrix)
        fit_kl = fit_params[0]
        fit_kl_array.append(fit_kl)
        fit_g1l = fit_params[1]
        fit_g1l_array.append(fit_g1l)
        fit_correc = fit_params[2]
        fit_correc_array.append(fit_correc)

        # Same again, using the best fitted network output data instead of the raw data
        print(f"k{l+1}: ", fit_kl, f"\ng[1]({l+1}): ", fit_g1l, f"\n1/nsq Correction Coeff: ", fit_correc)

        plt.clf()
        plt.scatter(inv_width_axis, l_fit_matrix, marker="x", label="Data Points")

        fitted_curve_y_array = []

        for i in x_axis:
            fitted_curve_y = zl_sq(i, fit_kl, fit_g1l, fit_correc)
            fitted_curve_y_array.append(fitted_curve_y)

        plt.plot(x_axis, fitted_curve_y_array, "r--", label="Best Fit")

        plt.legend()
        plt.xlabel(f'Inverse Width')
        plt.ylabel('<Zsq>')
        plt.suptitle(f'<Zsq> vs 1/w (Using Best Fitted OutDist Data) (Layer {l + 1})')
        plt.title(f'k({l + 1}) = {round(fit_kl, 2)}, g[1]({l + 1}) = {round(fit_g1l, 2)}, Correction Scale = {round(fit_correc, 2)}')
        os.makedirs(f'Plots\Analysis_Plots\{plot_type}\Avg_Zsq\Fitted_OutDist_Data_Plots\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}\{initial_width}w-{final_width}w & {depth}d', exist_ok=True)
        plt.savefig(f'Plots\Analysis_Plots\{plot_type}\Avg_Zsq\Fitted_OutDist_Data_Plots\Cb={Cb} Cw={Cw} Nb={Nboot} Mu={Mu}\{initial_width}w-{final_width}w & {depth}d\Layer {l + 1}.png')

    print("\n----Finished----")

    return


def avg_rho_sq(activation_function, k):

    coeff = 1 / (np.sqrt(2 * math.pi * k))
    integrand = lambda z: (np.exp((-1 / 2) * (1 / k) * (z ** 2))) * (activation_function(z) ** 2)
    integral = integrate.quad(integrand, -np.inf, np.inf)
    value = coeff * integral[0]
    est_error = integral[1]

    return value, est_error


def analytical_recursion(activation_function, X, depth, Cw, Cb):

    k0 = Cb + Cw*(X**2)
    ki = 0
    ki_array = []
    ki_err_array = []
    l_array = []

    for l in range(depth):

        if l == 0:
            ki = k0
        else:
            l_array.append(l + 1)
            avg_rho = avg_rho_sq(activation_function, ki)
            ki = (Cb + (Cw*(avg_rho[0])))
            ki_array.append(ki)
            ki_err = avg_rho[1]
            ki_err_array.append(ki_err)

        print(f"k{l+1} = ", ki)

    plt.clf()
    plt.plot(l_array, ki_array)
    plt.show()

    return