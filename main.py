import tensorflow as tf
import function as func

# Activation Function Options:
# [1.] tf.nn.tanh [2.] tf.nn.sigmoid [3.] tf.nn.elu [4.] tf.nn.relu [5.] tf.nn.selu [6.] tf.nn.gelu [7.] func.linear

####    func.initialise_network(activation_function, width, depth, Cw, Cb, Mu, Nboot, rescale)    ####
# Will instantiate and analyse a network "Nboot" times, plotting its output distributions layer by layer as well as the input weights and bias distributions.

#func.initialise_network(tf.nn.tanh, width=10, depth=10, Cw=1, Cb=0, Mu=0, Nboot=500, rescale=False)

#### func.numerical_analysis(activation_function, initial_width, final_width, depth, Cw, Cb, Mu, Nboot, rescale)
# Instantiated the network and completes a numerical analysis to give values for g0l, g1l and the O(1/n^2) corrections all as a function of width.

func.numerical_analysis(tf.nn.tanh, initial_width=10, final_width=15, depth=10, Cw=1, Cb=0, Mu=0, Nboot=1000, rescale=False)

#func.analytical_recursion(tf.nn.tanh, 100, 10, 1, 0)