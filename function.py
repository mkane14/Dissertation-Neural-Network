#This code contains everything to do with the neural network and extracting its statistics
import numpy as np
import tensorflow as tf
import plot

#Generates the weights and bias's for the network by drawing them at random from some normal distribution
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

            #Appending the output distribution for each layer of the network
            Layer_output_matrix[i, j] = Z

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


#This function initiates the neural network with the given inputs and gives the statistics of each step
def initialise_network(activation_function, width, depth, Cw, Cb, Mu, Nboot):

    #Input and output layer size (number of neurons)
    input_size = 1
    output_size = 1

    plot_type = plot.determine_plot_type(activation_function)

    #Prints the networks input statistics
    net_stat = f"Nboot={Nboot}, Cb={Cb} & Cw={Cw}, Mu={round(Mu, 2)}"
    print(f"Input Gaussian Values: {net_stat}")

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
    plot.bias_input_dist(Bhist, width, depth, Cw, Cb, Mu, Nboot, plot_type)

    #Plotting the weight input distribution
    plot.weight_input_dist(Whist, width, depth, Cw, Cb, Mu, Nboot, plot_type)

    # Input quantities
    X = tf.constant([[30]], dtype=tf.float32)

    #L'th layer output matrix
    Lom = np.zeros((Nboot, depth), dtype=object)

    #Output layer array
    output_array = []

    #Propagates the signal throughout the network for every instantiation
    for i in range(Nboot):
        Z = forward(X, i, activation_function, depth, Ws=Ws, Bs=Bs, Layer_output_matrix=Lom)
        output_array.append(Z.numpy()[0][0])

    #Plot the final layer output distribution
    ### FINAL OUTPUT LAYER IS PLOTTED IN "Out_Dist_Per_Layer" ###
    ### ONLY USEFUL IF YOU ONLY WANT THE FINAL LAYER'S OUTPUT DISTRIBUTION ###
    #plot.final_out_dist(output_array, width, depth, Cw, Cb, Mu, Nboot, plot_type)

    #Array to store FWHM values
    FWHM_array = []

    #Filtering the array data for the output distribution of each layer so we can plot it
    for i in range(depth):
        A = np.zeros(Nboot)
        for j in range(Nboot):
            matrLom = Lom[j, i]
            A[j] = matrLom[0, 0]

        #Plots the output distribution for every layer and appending each layers FWHM
        FWHM = plot.output_dist_per_layer(A, width, i, Cw, Cb, Mu, Nboot, plot_type)
        FWHM_array.append(FWHM)

    #Plotting the FWHM vs depth of the network
    plot.gaussian_width_and_depth(width, depth, Cw, Cb, Mu, Nboot, FWHM_array, plot_type)

    print("----Complete----")

    return