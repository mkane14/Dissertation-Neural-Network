import tensorflow as tf
import numerical
import analytical

# Activation Function Options:
# [1.] tf.nn.tanh [2.] tf.nn.sigmoid [3.] tf.nn.elu [4.] tf.nn.relu [5.] tf.nn.selu [6.] tf.nn.gelu [7.] func.linear

# Initiates the network and completes a numerical analysis
numerical.analysis(activation_function=tf.nn.tanh, input_x=0.1, initial_width=20, final_width=50, depth=50, Cw=1, Cb=0, Mu=0, Nboot=1000, rescale=False)

# Calculates and plots the analytical kernel recursion
#analytical.recursion(activation_function=tf.nn.tanh, input_x=0.1, depth=50, Cw=1, Cb=0)