import tensorflow as tf
import function as func

# Nboot value defines the number of initialisations of the network

# Activation Function Options: [1.] tf.nn.tanh [2.] tf.nn.sigmoid [3.] tf.nn.elu [4.] tf.nn.relu [5.] tf.nn.selu [6.] tf.nn.gelu

####    func.initialise_network(activation_function, width, depth, Nboot, STD, Mu)    ####

func.initialise_network(tf.nn.tanh, 10, 10, 1, 2, 0, 1000)
