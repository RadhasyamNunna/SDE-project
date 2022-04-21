import tensorflow.compat.v1 as tf
import numpy as np
import sys
from layers import *
from utils import *
tf.compat.v1.disable_eager_execution()


class Model(object):
    def __init__(self, config, dir_output,name):
        self._config = config
        self._dir_output = dir_output
        self.name=name
        tf.reset_default_graph() 
        tf.set_random_seed(1)  


    def training(self):
        self.X = tf.compat.v1.placeholder("float", [None, self._config['num_input']])
        self.Y = tf.compat.v1.placeholder("float", [None, 1])
        self.lr = tf.compat.v1.placeholder("float") 
        self.output = neural_net(self.X,
                                 self._config['num_layer'],
                                 self._config['num_neuron'],
                                 self._config['lambda'],self.name)
        l2_loss = tf.losses.get_regularization_loss()
        self.loss = l2_loss + tf.losses.mean_squared_error(self.Y, self.output)
        
        self.train_optimizer(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def validation(self):
        self.X = tf.compat.v1.placeholder("float", [None, self._config['num_input']])
        self.Y = tf.compat.v1.placeholder("float", [None, 1])
        self.lr = tf.compat.v1.placeholder("float") 
        self.output = neural_net(self.X,
                                 self._config['num_layer'],
                                 self._config['num_neuron'],
                                 self._config['lambda'],self.name)
        l2_loss = tf.losses.get_regularization_loss()
        self.loss = l2_loss + tf.losses.mean_squared_error(self.Y, self.output)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads, vs     = zip(*optimizer.compute_gradients(loss))
            grads, gnorm  = tf.clip_by_global_norm(grads, 1)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))



    def train(self, X_matrix, perf_value, lr_initial):
        lr = lr_initial
        decay = lr_initial/1000

        m = X_matrix.shape[0]
        batch_size = m
        seed = 0    
        for epoch in range(1, 2000):

            minibatch_loss = 0
            num_minibatches = int(m/batch_size)
            seed += 1
            minibatches = random_mini_batches(X_matrix, perf_value, batch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, t_l, pred = self.sess.run([self.train_op, self.loss, self.output],
                                             {self.X : X_matrix, self.Y: perf_value, self.lr: lr})
                minibatch_loss += t_l/num_minibatches

            if epoch % 500 == 0 or epoch == 1:

                rel_error = np.mean(np.abs(np.divide(perf_value.ravel() - pred.ravel(), perf_value.ravel())))
                if self._config['verbose']:
                    print("     cost function: {:.4f}".format(minibatch_loss))
                    print("     train relative error: {:.4f}".format(rel_error))

            lr = lr*1/(1 + decay*epoch)


    def model_saving(self):
        dir_model = self._dir_output + "model.weights/"
        init_dir(dir_model)
        sys.stdout.write("\r- Saving model...")
        sys.stdout.flush()
        self.saver.save(self.sess, dir_model + 'model.ckpt')
        sys.stdout.write("\r")
        sys.stdout.flush()
        self.logger.info("- Saved model in {}".format(dir_model))


    def predict(self, X_matrix_pred):
        Y_pred_val = self.sess.run(self.output, {self.X: X_matrix_pred})
        return Y_pred_val
