import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
import numpy as np
import sys
tf.compat.v1.disable_eager_execution()

def neural_net(tf_x, n_layer, n_neuron, lambd,name):
    layer = tf_x
    for i in range(1, n_layer+1):
        if (i==1 and name=='sparse'):
            layer = tf.keras.layers.Dense( n_neuron, tf.nn.relu,
                                    kernel_initializer=tf.initializers.glorot_normal(seed=1),
                                    kernel_regularizer=tf.keras.regularizers.L1(float(lambd)))(layer)
        else:
            layer = tf.keras.layers.Dense( n_neuron, tf.nn.relu,
                                    kernel_initializer=tf.initializers.glorot_normal(seed=1))(layer)
    output =tf.keras.layers.Dense(1)(layer)
    return output