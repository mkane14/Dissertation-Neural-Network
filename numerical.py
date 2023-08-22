#This code contains everything to do with the neural network and extracting its statistics
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import plot
from scipy.optimize import curve_fit
from scipy import integrate
import os
import analytical

#### INITIATE THE NETWORK USING THE MAIN FILE AND CALLING THE NUMERICAL ANALYSIS FUNCTION ####

#Generates the weights and bias's for the network by drawing them at random from some normal distribution
def initialization(input_size_, hidden_size_, Mu_, STD_, nl, weightorbias):

    if weightorbias == "weight":
        Wname = 'W{}'.format(nl)
        out = tf.Variable(tf.random.normal([input_size_, hidden_size_], mean=Mu_, stddev=STD_), name=Wname)

    if weightorbias == "bias":
        bname = 'b{}'.format(nl)
        out = tf.Variable(tf.random.normal([hidden_size_], mean=Mu_, stddev=STD_), name=bname)

    return out


# This function propagates the signal through the network
def forward(X, i, activation_function, width_, depth_, Ws, Bs, Layer_out_matrix, Layer_zsq_matrix, Layer_z4_matrix):

    with tf.GradientTape() as tape:
        tape.watch([param for param in [Ws[i, j] for j in range(depth_)] + [Bs[i, j] for j in range(depth_)]])

        # Setting the preactivation of the first hidden layer as the input data set
        A = X

        for l in range(depth_):

            #print(f"Activations of {l}:\n{tf.get_static_value(A[0])}\n")
            layer_width = width_[l+1]
            # Calculating the preactivation and activation function for a given hidden layer
            Z = tf.add(tf.matmul(A, Ws[i, l]), Bs[i, l])
            A = activation_function(Z)

            zsq = 0
            z4 = 0

            #print(f"Bias: {Bs[i, l]}")
            #print(f"\nWeights: {Ws[i, l]}\n")
            #print(f"Z{l+1} of boot {i+1}:\n{tf.get_static_value(Z[0])}")

            for k in tf.get_static_value(Z[0]):
                z = k
                #print("z", z)
                zsq += (z)**2
                #print("zsq", zsq)
                z4 += (z)**4
                #print("z4", z4)

            Layer_zsq_matrix[i, l] = zsq/layer_width  ### THESE ARE AVERAGES OVER THE NEURONS
            Layer_z4_matrix[i, l] = z4/layer_width  ### THESE ARE AVERAGES OVER THE NEURONS

            # Appending the output distribution for each layer of the network - not an average
            Layer_out_matrix[i, l] = Z

        #print("next")
        Z_last = Z

    return Z_last


# Used to exract data from the weights and bias generation, allows for histogram plot
def bootobj_extr(layer_, el1_, el2_, Nboot, Ws, Bs):

    A = np.zeros(Nboot)
    B = np.zeros(Nboot)

    for i in range(Nboot):
        matrW = Ws[i, layer_]
        A[i] = matrW[el1_, el2_]
        vecB = Bs[i, layer_]
        B[i] = vecB[el1_]

    return A, B


# Linear activation function
def linear(Z):
    A = Z
    return A


def sin(Z):
    A = np.sin(Z)
    return A


# <Zsq> formula to use with curve_fit and the data
def zl_sq(w, g0l, g1l, correc):
    return g0l + (g1l * w) + (correc * (w**2))


def vl(w, vl, correc):
    return (vl * w) + (correc * (w**2))


# Formula to calculate the error from the standard deviation
def standard_error(sigma, N, width):
    std_error = sigma / (np.sqrt(N)*width)
    return std_error


#This function initiates the neural network with the given inputs and gives the statistics of each step
def initialise_network(activation_function, input_x, width, depth, Cw, Cb, Mu, Nboot, rescale):

    print("\n----Initiating----")

    #Input and output layer size (number of neurons)
    input_size = 1
    output_size = 1

    plot_type = plot.determine_plot_type(activation_function)

    #Prints the networks input statistics
    net_stat = f"Nboot={Nboot}, Cb={Cb} & Cw={Cw}, Mu={round(Mu, 2)}"
    print(f"Input Gaussian Values: {net_stat}")

    #Creates an array with sizes of each layer as each value
    # lsizes = layer sizes
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
                Ws[i, l] = initialization(lsizes[l], lsizes[l + 1], Mu, np.sqrt(Cw/lsizes[l]), l, weightorbias="weight")

    #Filtering the input bias and weight distribution data to plot on a histogram
    Whist, Bhist = bootobj_extr(1, 1, 1, Nboot=Nboot, Ws=Ws, Bs=Bs)

    #Plotting the bias input distribution
    plot.bias_input_dist(input_x, Bhist, width, depth, Cw, Cb, Mu, Nboot, plot_type)

    #Plotting the weight input distribution
    plot.weight_input_dist(input_x, Whist, width, depth, Cw, Cb, Mu, Nboot, plot_type)

    # Input quantities
    X = tf.constant([[input_x]], dtype=tf.float32)

    #L'th layer output matrix
    Lom = np.zeros((Nboot, depth), dtype=object)    #purely for the output distribution
    Lzsqm = np.zeros((Nboot, depth), dtype=object)  #storing <z^2>
    Lz4m = np.zeros((Nboot, depth), dtype=object)   #storing <z^4>

    #Final layer output layer array
    output_array = []

    #Propagates the signal throughout the network for every instantiation
    for i in range(Nboot):
        Z_array = forward(X, i, activation_function, lsizes, depth, Ws=Ws, Bs=Bs, Layer_out_matrix=Lom, Layer_zsq_matrix=Lzsqm, Layer_z4_matrix=Lz4m)
        output_array.append(Z_array.numpy()[0][0])

    #Plot the final layer output distribution
    plot.final_out_dist(output_array, input_x, width, depth, Cw, Cb, Mu, Nboot, plot_type, rescale)

    #Arrays to store values for FWHM of fitted data
    Var_fit_array = []
    Var_fit_err_array = []
    Var_data_array = []
    Var_data_err_array = []

    zsq_array = []
    zsq_err_array = []
    zdiff_array = []
    zdiff_err_array = []

    #Filtering the array data for the output distribution of each layer so we can plot it
    for l in range(depth):

        Z = np.zeros(Nboot)

        zsq = 0
        zsq_std_array = []

        z4 = 0
        z4_std_array = []

        for i in range(Nboot):
            matrLom = Lom[i, l]
            Z[i] = matrLom[0, 0]
            # Calculating <Zsq> directly from the output data for every layer
            zsq += Lzsqm[i, l] / (Nboot)
            zsq_std_array.append(zsq)
            z4 += Lz4m[i, l] / (Nboot)
            z4_std_array.append(z4)

        #Plots the output distribution for every layer and appending each layers FWHM
        FWHM, STD_fit, STD_fit_err, STD_data, STD_data_err = plot.output_dist_per_layer(Z, input_x, width, depth, l, Cw, Cb, Mu, Nboot, plot_type, rescale)

        
        Var_fit_array.append(STD_fit**2)
        Var_fit_err_array.append(STD_fit_err)
        Var_data_array.append(STD_data**2)
        Var_data_err_array.append((STD_data_err))

        # Appending zsq = <Z^2> and calculating zdiff = 1/3(<Z^4> - 3<Z^2>^2)
        zsq_array.append(zsq)
        zsq_sq = zsq**2
        zdiff = (1/3) * (z4 - (3*zsq_sq))
        zdiff_array.append(zdiff)

        zsq_std = plot.std_dev(zsq_std_array)
        zsq_err = standard_error(zsq_std, Nboot, lsizes[l+1])
        zsq_err_array.append(zsq_err)

        zsq_err1 = standard_error(zsq_std, Nboot, lsizes[l+1])
        zsq_sq_err = zsq_sq * np.sqrt(2 * ((zsq_err1 / zsq) ** 2))
        z4_std = plot.std_dev(z4_std_array)
        z4_err = standard_error(z4_std, Nboot, lsizes[l+1])

        zdiff_err = np.sqrt((z4_err ** 2) + (zsq_sq_err ** 2))
        zdiff_err_array.append(zdiff_err)

        #print("Layer", l+1)
        #print("Inv. Width ", 1 / lsizes[l+1])
        #print("zsq: ", zsq)
        #print("zsq_sq: ", zsq_sq)
        #print("z4: ", z4)
        #print("zdiff: ", zdiff)
        #print("STD: ", STD_data)
        #print("zsq err: ", zsq_err)
        #print("zsq_sq err: ", zsq_sq_err)
        #print("z4 err: ", z4_err)
        #print("z diff err: ", zdiff_err)

    print("----Complete----")

    return zsq_array, zsq_err_array, zdiff_array, zdiff_err_array, Var_data_array, Var_data_err_array, Var_fit_array, Var_fit_err_array, lsizes

# Function to determine g0l (g0l), g1l and the order(1/nsq) correction to eq's 5.108-5.110 for finite width networks for a single input
def analysis(activation_function, input_x, initial_width, final_width, depth, Cw, Cb, Mu, Nboot, rescale):

    analytical_x_axis = range(1, depth + 1)
    analytical_y_axis = analytical.recursion(activation_function, input_x, depth, Cw, Cb)

    # Determines plot from activation function - just for plotting and saving
    plot_type = plot.determine_plot_type(activation_function)

    # The difference in the initial and final width used to define the dimensions of the arrays
    width_diff = (final_width - initial_width) + 1
    x_axis = np.linspace(1/initial_width, 1/final_width, 500)

    # Defining the matrices used to store values for different architectures the network is run over
    zsq_matrix = np.zeros((depth, width_diff), dtype=object)
    zsq_err_matrix = np.zeros((depth, width_diff), dtype=object)
    zdiff_matrix = np.zeros((depth, width_diff), dtype=object)
    zdiff_err_matrix = np.zeros((depth, width_diff), dtype=object)

    # Data is using the raw output through signal propagation from layer to layer - actual network output
    g0l_array = []
    g0l_err_array = []

    g1l_array = []
    g1l_err_array = []

    zsq_correc_array = []
    zsq_correc_err_array = []

    vl_array = []
    vl_err_array = []

    vl_correc1_array = []
    vl_correc1_err_array = []

    residual_y = []
    residual_y_error = []

    # Initiating the network for the variety of widths defined in order to allow for plots
    for w, j in enumerate(range(initial_width, final_width+1)):

        # Storing the network outputs for every width
        zsq_data, zsq_err_data, zdiff_data, zdiff_err_data, Var_data, Var_data_err, Var_fit, Var_fit_err, lsizes = initialise_network(activation_function, input_x, j, depth, Cw, Cb, Mu, Nboot, rescale)

        plt.clf()
        plt.scatter(range(1, depth+1), Var_data, color="r", marker="x", label="Data")
        plt.errorbar(range(1, depth+1), Var_data, ecolor="r", xerr=None, yerr=Var_data_err, ls="None")
        plt.scatter(range(1, depth+1), Var_fit, color="k", marker=".", label="Fit")
        plt.errorbar(range(1, depth + 1), Var_fit, ecolor="k", xerr=None, yerr=Var_fit_err, ls="None")
        plt.plot(analytical_x_axis, analytical_y_axis, "m--", label="Analytical Recursion")
        plt.legend()
        plt.xlabel("Layer [l]")
        plt.ylabel("L'th Output Dist. Variance ($K^{(l)}$)")
        plt.title(f"Output Dist. Variance vs Layer (For $n_l$={j})")
        os.makedirs(f"Output Dists\Variance vs Layer\{plot_type}\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x = {input_x}\{initial_width}w-{final_width}w & {depth}d", exist_ok=True)
        plt.savefig(f"Output Dists\Variance vs Layer\{plot_type}\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x = {input_x}\{initial_width}w-{final_width}w & {depth}d\Width {j}.png", bbox_inches="tight")
        #plt.show()
        plt.clf()


        diff = 0
        residual_err = 0

        for u in range(depth):
            diff += analytical_y_axis[u] - Var_data[u]
            residual_err += np.sqrt(Var_fit_err[u]**2 + Var_data_err[u]**2)

        residual_y.append(diff)
        residual_y_error.append(residual_err)

        # Storing the network outputs for every layer
        for l in range(depth):
            zsq_matrix[l, w] = zsq_data[l]
            zsq_err_matrix[l, w] = zsq_err_data[l]
            zdiff_matrix[l, w] = zdiff_data[l]
            zdiff_err_matrix[l, w] = zdiff_err_data[l]

    # Starting the numerical analysis
    for l in range(depth):

        print(f"\n----Numerical Analysis for Layer {l + 1}----")
        plt.clf()

        # Arrays to store values l'th layer values for single plot at a time analysis
        zsq_l_array = []
        zsq_err_l_array = []

        zdiff_l_array = []
        zdiff_err_l_array = []

        inv_width_axis = []

        if l <= depth-2:

            #Picking the specific values required from the matrices that allow for us to plot <Zsq> against 1/w
            for w, j in enumerate(range(initial_width, final_width+1)):
                zsq_l_array.append(zsq_matrix[l, w])
                zsq_err_l_array.append(zsq_err_matrix[l, w])
                inv_width_axis.append(1/j)

            # Best fitting the stored values in order to match "zl_sq", this gives us the best values for g0l, g1l and the correction using the raw network output

            zsq_params, zsq_pcov = curve_fit(zl_sq, inv_width_axis, zsq_l_array, sigma=zsq_err_l_array, maxfev=5000, bounds=((0, -np.inf, -np.inf), (np.inf, np.inf, np.inf)))

            g0l = zsq_params[0]
            g01_err = np.sqrt(np.diag(zsq_pcov)[0])
            g0l_array.append(g0l)
            g0l_err_array.append(g01_err)

            g1l = zsq_params[1]
            g1l_err = np.sqrt(np.diag(zsq_pcov)[1])
            g1l_array.append(g1l)
            g1l_err_array.append(g1l_err)

            zsq_correc = zsq_params[2]
            zsq_correc_err = np.sqrt(np.diag(zsq_pcov)[2])
            zsq_correc_array.append(zsq_correc)
            zsq_correc_err_array.append(zsq_correc_err)

            # Printing these values for the specific layers
            print(f"G[0]({l + 1}): {g0l} +/- {g01_err}", f"\nG[1]({l + 1}): {g1l} +/- {g1l_err}", f"\nCorrection: {zsq_correc} +/- {zsq_correc_err}")

            # Using the best fit parameters to make a best fit curve plot, allows for visualisation of how well the g0l, g1l and corrections fit the data
            y_curve_array = []

            for i in x_axis:
                y_curve = zl_sq(i, g0l, g1l, zsq_correc)
                y_curve_array.append(y_curve)

            # Plotting the best fitted curve for the data
            plt.clf()

            label = str("Best Fit:\n$G^{{0}(l)}$") + str(f" = {round(g0l, 3)}$\pm${round(g01_err, 3)}\n") + str("$G^{{1}(l)}$") + str(f" = {round(g1l, 3)}$\pm${round(g1l_err, 3)}") + str("\nO($\dfrac{1}{n^2}$)") + str(f"={round(zsq_correc, 3)}$\pm${round(zsq_correc_err, 3)}")
            plt.plot(x_axis, y_curve_array, "r--", label=label)

            # Now plotting the stored network outputs as a scatter graph: <Zsq> vs 1/n
            plt.scatter(inv_width_axis, zsq_l_array, c="k", marker=".", label=f"Data")
            plt.errorbar(inv_width_axis, zsq_l_array, ecolor="k", xerr=None, yerr=zsq_err_l_array, ls="None")

            # Properly labelling and saving the associated plot
            plt.legend()
            plt.xlabel('Inv. Width [$\dfrac{1}{n_{(l)}}$]')
            plt.ylabel('$<Z^2>$')
            #plt.suptitle('$<Z^2>$ vs $dfrac{1}{n_{(l)}$')
            plt.title(f"Layer {l + 1} for Input x = {input_x}")
            os.makedirs(f'Numerical Plots\{plot_type}\Zsq vs Inv.Width\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x={input_x}\{initial_width}w-{final_width}w & {depth}d', exist_ok=True)
            plt.savefig(f'Numerical Plots\{plot_type}\Zsq vs Inv.Width\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x={input_x}\{initial_width}w-{final_width}w & {depth}d\Layer {l + 1}.png', bbox_inches="tight")
            #plt.show()

        if l == depth - 1:
            print(f"G[0](l) and G[1](l) values cannot be extracted for the final layer (n({l+1}) = {lsizes[l+1]})")

        if l == 0:
            print(f"V(l) values cannot be extracted for l={l+1} (n({l}) = {lsizes[l]})")

        inv_width_axis = []

        if l >= 1:

            for w, j in enumerate(range(initial_width, final_width+1)):
                zdiff_l_array.append(zdiff_matrix[l, w])
                zdiff_err_l_array.append(zdiff_err_matrix[l, w])
                inv_width_axis.append(1 / j)

            # Plotting the graph for the vl values plot
            vl_params, vl_pcov = curve_fit(vl, inv_width_axis, zdiff_l_array, sigma=zdiff_err_l_array, maxfev=5000)

            vl_value = vl_params[0]
            vl_value_err = np.sqrt(np.diag(vl_pcov)[0])
            vl_array.append(vl_value)
            vl_err_array.append(vl_value_err)

            vl_correc1 = vl_params[1]
            vl_correc1_err = np.sqrt(np.diag(vl_pcov)[1])
            vl_correc1_array.append(vl_correc1)
            vl_correc1_err_array.append(vl_correc1_err)

            print(f"V({l + 1}): {vl_value} +/- {vl_value_err}", f"\n1/(n^2) Correction: {vl_correc1} +/- {vl_correc1_err}")

            y_curve_array = []

            for j in x_axis:
                y_curve = vl(j, vl_value, vl_correc1)
                y_curve_array.append(y_curve)

            plt.clf()
            plt.scatter(inv_width_axis, zdiff_l_array, color="k", marker=".", label="Data")
            plt.errorbar(inv_width_axis, zdiff_l_array, ecolor="k", xerr=None, yerr=zdiff_err_l_array, ls="None")
            label = str("Best Fit:\n") + str("$V_{(l)}$") + str(f" = {round(vl_value, 4)}$\pm${round(vl_value_err, 4)}\n") + str("O($\dfrac{1}{n^2}$)") + str(f" = {round(vl_correc1, 4)}$\pm${round(vl_correc1_err, 4)}")
            plt.plot(x_axis, y_curve_array, "r--", label=label)
            plt.legend()
            plt.xlabel('$\dfrac{1}{n_{(l-1)}}$')
            plt.ylabel('$\dfrac{1}{3}<z^4> - 3<Z^2>^2)$')
            #plt.suptitle(f'Zdiff vs 1/n(l-1)')
            plt.title(f"Layer {l + 1} for Input x = {input_x}")
            os.makedirs(f'Numerical Plots\{plot_type}\Zdiff vs Inv.Width\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x={input_x}\{initial_width}w-{final_width}w & {depth}d', exist_ok=True)
            plt.savefig(f'Numerical Plots\{plot_type}\Zdiff vs Inv.Width\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x={input_x}\{initial_width}w-{final_width}w & {depth}d\Layer {l + 1}.png', bbox_inches="tight")
            #plt.show()


    plt.clf()
    plt.scatter(range(initial_width, final_width+1), residual_y, marker="x", color="k", label="Fit-Data Residuals")
    plt.errorbar(range(initial_width, final_width+1), residual_y, xerr=None, yerr=residual_y_error, ecolor="k", ls="None")
    plt.legend()
    plt.ylabel("Residual Value")
    plt.xlabel("Width [$n_l$]")
    os.makedirs(f'Output Dists\Residual Plots\{plot_type}\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x={input_x}',exist_ok=True)
    plt.savefig(f"Output Dists\Residual Plots\{plot_type}\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x={input_x}\{initial_width}w-{final_width}w & {depth}d.png", bbox_inches="tight")


    plt.clf()
    plt.scatter(range(1, depth), g0l_array, marker="x", c="k", label="$G^{{0}(l)}$")
    plt.plot(analytical_x_axis, analytical_y_axis, "r--", label="Analytical Recursion")
    plt.errorbar(range(1, depth), g0l_array, xerr=None, ecolor="k", yerr=g0l_err_array, ls="None")
    plt.legend()
    # plt.suptitle("$G^{{0}(l)}$ vs Layer")
    plt.title(f"Evaluated over {initial_width}w-{final_width}w for {depth}d & Input x = {input_x}")
    plt.xlabel("Layer [l]")
    plt.ylabel("$G^{{0}(l)}$")
    os.makedirs(f'Numerical Plots\{plot_type}\G[0](l) vs Layer\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x={input_x}', exist_ok=True)
    plt.savefig(f'Numerical Plots\{plot_type}\G[0](l) vs Layer\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x={input_x}\{initial_width}w-{final_width}w & {depth}d.png', bbox_inches="tight")
    # plt.show()

    plt.clf()
    plt.scatter(range(1, depth), g1l_array, marker="x", c="k", label="$G^{{1}(l)}$")
    plt.errorbar(range(1, depth), g1l_array, xerr=None, ecolor="k", yerr=g1l_err_array, ls="None")
    plt.legend()
    #plt.suptitle("G{1}(l) vs Layer")
    plt.title(f"Evaluated over {initial_width}w-{final_width}w for {depth}d & Input x = {input_x}")
    plt.xlabel("Layer [l]")
    plt.ylabel("$G^{1(l)}$")
    os.makedirs(f'Numerical Plots\{plot_type}\G[1](l) vs Layer\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x={input_x}', exist_ok=True)
    plt.savefig(f'Numerical Plots\{plot_type}\G[1](l) vs Layer\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x={input_x}\{initial_width}w-{final_width}w & {depth}d.png', bbox_inches="tight")
    #plt.show()

    plt.clf()
    plt.scatter(range(2, depth+1), vl_array, marker="x", c="k", label="$V^{(l)}$")
    plt.legend()
    plt.errorbar(range(2, depth+1), vl_array, ecolor="k", xerr=None, yerr=vl_err_array, ls="None")
    #plt.suptitle(f"V(l) vs Layer")
    plt.title(f"Evaluated over {initial_width}w-{final_width}w for {depth}d & Input x = {input_x}")
    plt.xlabel("Layer [l]")
    plt.ylabel("$V^{(l)}$")
    os.makedirs(f'Numerical Plots\{plot_type}\V(l) vs Layer\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x={input_x}', exist_ok=True)
    plt.savefig(f'Numerical Plots\{plot_type}\V(l) vs Layer\Cb={Cb} Cw={round(Cw, 3)} Nb={Nboot}\ x={input_x}\{initial_width}w-{final_width}w & {depth}d.png', bbox_inches="tight")
    #plt.show()
    plt.clf()

    print("\n----Finished----")

    return