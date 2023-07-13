import tensorflow as tf
import Test
import function as func
import vary

####    vary.network_architecture(variable, activation_function, Nboot, Mu)    ####
####    vary.network_statistics(variable, activation_function, width, depth)    ####
####    func.plot_activation_hist(activation_function, width, depth, Nboot, STD, Mu)    ####

#Nboot value defines the number of initialisations of the network

# Vary Architecture Options: [1.] "width" [2.] "depth" [3.] "width, depth" <----> USE QUOTATION MARKS "......" FOR INPUT

# Vary Statistics Options: [1.] "Nboot" [2.] "STD" [3.] "Mu" <----> USE QUOTATION MARKS "......" FOR INPUT

# Activation Function Options: [1.] tf.nn.tanh [2.] tf.nn.sigmoid [3.] tf.nn.elu [4.] tf.nn.relu [5.] tf.nn.selu [6.] tf.nn.gelu

#vary.network_architecture(variable="width, depth", activation_function=tf.nn.relu, Nboot=1000, Mu=0.5)

#func.plot_activation_hist(tf.nn.tanh, 3, 5, 1000, 1, 2, 0)
#function.plot_activation_hist(tf.nn.tanh, 10, 15, 1000, 0.1, 0)

#vary.network_architecture(variable="depth", activation_function=tf.nn.tanh, Nboot=1000, Mu=0)
func.initialise_network(tf.nn.tanh, 10, 10, 1000, 1, 2, 0)
#Test.initialise_network(tf.nn.relu, 10, 10, 1000, 1, 2, 0)

#vary.network_statistics(variable= , activation_function= , width= , depth= )