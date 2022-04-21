import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import time
import argparse
tf.compat.v1.disable_eager_execution()
from model import Model
import time
import argparse

def nn_l1_val(X_train1, Y_train1, X_train2, Y_train2, n_layer, lambd, lr_initial):
    config = dict()
    config['num_input'] = X_train1.shape[1]
    config['num_layer'] = n_layer
    config['num_neuron'] = 128
    config['lambda'] = lambd
    config['verbose'] = 0
    dir_output = './output'
    model = Model(config, dir_output,'sparse')
    model.training()
    model.train(X_train1, Y_train1, lr_initial)
    Y_pred_val = model.predict(X_train2)
    abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
    rel_error = np.mean(np.abs(np.divide(Y_train2 - Y_pred_val, Y_train2)))
    return abs_error, rel_error


def system_samplesize(sys_name):
    return softs[sys_name]


def seed_generator(sys_name, sample_size):
    N_train_all = system_samplesize(sys_name)
    if sample_size in N_train_all:
        seed_o = np.where(N_train_all == sample_size)[0][0]
    else:
        seed_o = np.random.randint(1, 101)
    return seed_o

softs={}
softs['Apache']=np.multiply(9, [1, 2, 3, 4, 5])
softs['BDBJ']=np.multiply(26, [1, 2, 3, 4, 5])
softs['BDBC']=np.multiply(18, [1, 2, 3, 4, 5])
softs['LLVM']=np.multiply(11, [1, 2, 3, 4, 5])
softs['SQL']=np.multiply(39, [1, 2, 3, 4, 5])
softs['x264']=np.multiply(16, [1, 2, 3, 4, 5])
softs['Dune']=np.asarray([49, 78, 240, 375]) 
softs['hipacc']= np.asarray([261, 736, 528, 1281]) 
softs['hsmgp']= np.asarray([77, 173, 384, 480]) 
softs['javagc']= np.asarray([423, 534, 855, 2571]) 
softs['sac']= np.asarray([2060, 2295, 2499, 3261])
