import tensorflow as tf
import function as func

# Activation Function Options: [1.] tf.nn.tanh [2.] tf.nn.sigmoid [3.] tf.nn.elu [4.] tf.nn.relu [5.] tf.nn.selu [6.] tf.nn.gelu

####    func.initialise_network(activation_function, width, depth, Cb, Cw, Mu, Nboot)    ####

func.initialise_network(tf.nn.tanh, 5, 15, 2, 1, 0, 500)