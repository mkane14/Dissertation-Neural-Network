import tensorflow as tf
import function as func

# Activation Function Options: [1.] tf.nn.tanh [2.] tf.nn.sigmoid [3.] tf.nn.elu [4.] tf.nn.relu [5.] tf.nn.selu [6.] tf.nn.gelu

####    func.initialise_network(activation_function, width, depth, Nboot, Cb, Cw, Mu, Nboot)    ####

func.initialise_network(activation_function=tf.nn.tanh, width=10, depth=10, Cb=1, Cw=2, Mu=0, Nboot=1000)
