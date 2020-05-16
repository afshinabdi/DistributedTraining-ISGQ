"""
   Convolutional neural network for classification of CIFAR10 data.
   The default is Lenet-5 like structure, two convolutional layers, followed by two fully connected ones.
   The filters' shapes are:
   [5, 5, 1, 32], [5, 5, 32, 64], [7 * 7 * 64, 384], [384, 192], [192, 10]
"""

import itertools
import numpy as np
import scipy.stats as st
import tensorflow as tf


class AlexnetModel:
    def __init__(self):
        self._image_size = 227
        self._num_layers = 8
        self._num_convlayers = 5
        self._num_fclayers = 3
        self._sess = None
        self._global_counter = None

        # parameters of the neural network
        self._drop_rate = None
        self._input = None
        self._output = None
        self._logit = None
        self._target = None

        self._nn_weights = []
        self._nn_biases = []

        # evaluation and training of the neural network
        self._accuracy = None
        self._optimizer = None
        self._trainOp = None
        self._learning_rate = 0.01
        self._loss = None

        # gradients of the parameters
        self._grad_W = None
        self._grad_b = None

        # to apply externally computed gradients
        self._input_gW = None
        self._input_gb = None
        self._apply_gradients = None

        # forward and backward signals in the neural network
        self._fw_signals = None
        self._bp_signals = None

    # =========================================================================
    # build the neural network
    def create_network(self, parameters: dict):
        if parameters.get('initial_w') is None:
            initial_weights, initial_biases = self._generate_random_parameters()
        else:
            initial_weights = parameters.get('initial_w')
            initial_biases = parameters.get('initial_b')

        graph = tf.Graph()
        with graph.as_default():
            # 1- create the neural network with the given/random initial weights/biases
            self._create_initialized_network(initial_weights, initial_biases)

            # 2- if required, add regularizer to the loss function
            l1 = parameters.get('l1_regularizer')
            if l1 is not None:
                self._add_l1regulizer(w=l1)

            l2 = parameters.get('l2_regularizer')
            if l2 is not None:
                self._add_l2regulizer(w=l2)

            # 3- if requried, add the training algorithm
            alg = parameters.get('training_alg')
            if alg is not None:
                self._add_optimizer(parameters)

                # 4- compute gradients?
                if parameters.get('compute_gradients', False):
                    self._add_gradient_computations()

            # 4- add gradient quantization
            if parameters.get('quantizer', False):
                self._add_quantizer(parameters)

            initializer = tf.global_variables_initializer()

        self._sess = tf.Session(graph=graph)
        self._sess.run(initializer)

    def _add_convolution_layer(self, x, kernel, bias, strides, padding):
        output = tf.nn.conv2d(x, kernel, strides=strides, padding=padding)
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

        return output

    def _add_splitted_convolutional_layer(self, x, kernel, bias, strides, padding):
        h0, h1 = tf.split(kernel, 2, axis=3)
        x0, x1 = tf.split(x, 2, axis=3)

        x0 = tf.nn.conv2d(x0, h0, strides=strides, padding=padding)
        x1 = tf.nn.conv2d(x1, h1, strides=strides, padding=padding)

        output = tf.concat([x0, x1], axis=3)
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

        return output

    def _add_fully_connected_layer(self, x, weight, bias, func=''):        
        x = tf.nn.dropout(x, rate=self._drop_rate)
        y = tf.matmul(x, weight) + bias
        self._fw_signals += [x]
        self._bp_signals += [y]

        if func == 'relu':
            output = tf.nn.relu(y)
        elif func == 'softmax':
            self._logit = y
            output = tf.nn.softmax(y)
        else:
            output = y

        return output

    def _add_max_pooling(self, x):
        output = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        return output

    # create neural network with random initial parameters
    def _generate_random_parameters(self):
        layer_shapes = [
            [11, 11, 3, 96], [5, 5, 48, 256], [3, 3, 256, 384], [3, 3, 192, 384], [3, 3, 192, 256], [9216, 4096],
            [4096, 4096], [4096, 1000]
        ]

        num_layers = len(layer_shapes)

        initial_weights = [0] * num_layers
        initial_biases = [0] * num_layers
        # create initial parameters for the network
        for n in range(num_layers):
            initial_weights[n] = st.truncnorm(-2, 2, loc=0, scale=0.1).rvs(layer_shapes[n])
            initial_biases[n] = np.ones(layer_shapes[n][-1]) * 0.1

        return initial_weights, initial_biases

    # create a convolutional neural network with given initial parameters
    def _create_initialized_network(self, initial_weights, initial_biases):
        self._fw_signals = []
        self._bp_signals = []
        self._nn_weights = []
        self._nn_biases = []

        # create weights and biases of the neural network
        name_scopes = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']
        for layer, init_w, init_b in zip(itertools.count(), initial_weights, initial_biases):
            with tf.variable_scope(name_scopes[layer]):
                w = tf.Variable(init_w.astype(np.float32), dtype=tf.float32, name='weights')
                b = tf.Variable(init_b.astype(np.float32), dtype=tf.float32, name='biases')

            self._nn_weights += [w]
            self._nn_biases += [b]

        self._input = tf.placeholder(tf.float32, shape=[None, self._image_size, self._image_size, 3])
        self._target = tf.placeholder(tf.int32, shape=None)
        self._drop_rate = tf.placeholder(tf.float32, name='drop-out')

        x = self._input

        # 1- convolution, local response normalization, and max-pooling
        x = self._add_convolution_layer(x, self._nn_weights[0], self._nn_biases[0], [1, 4, 4, 1], 'SAME')
        x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
        x = self._add_max_pooling(x)

        # 2- splitted convolution, local response normalization, and max-pooling
        x = self._add_splitted_convolutional_layer(x, self._nn_weights[1], self._nn_biases[1], [1, 1, 1, 1], 'SAME')
        x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
        x = self._add_max_pooling(x)

        # 3- only a convolution
        x = self._add_convolution_layer(x, self._nn_weights[2], self._nn_biases[2], [1, 1, 1, 1], 'SAME')

        # 4- splitted convolution
        x = self._add_splitted_convolutional_layer(x, self._nn_weights[3], self._nn_biases[3], [1, 1, 1, 1], 'SAME')

        # 5- splitted convolutional layer, max-pooling
        x = self._add_splitted_convolutional_layer(x, self._nn_weights[4], self._nn_biases[4], [1, 1, 1, 1], 'SAME')
        x = self._add_max_pooling(x)

        # 6- fully connected layer, relu
        x = tf.reshape(x, [-1, initial_weights[5].shape[0]])
        x = tf.nn.dropout(x, rate=self._drop_rate)
        x = self._add_fully_connected_layer(x, self._nn_weights[5], self._nn_biases[5], func='relu')

        # 7- another fully connected layer, relu
        x = tf.nn.dropout(x, rate=self._drop_rate)
        x = self._add_fully_connected_layer(x, self._nn_weights[6], self._nn_biases[6], func='relu')

        # 8- output fully connected layer, softmax
        x = tf.nn.dropout(x, rate=self._drop_rate)
        self._output = self._add_fully_connected_layer(x, self._nn_weights[7], self._nn_biases[7], func='softmax')


        # loss function
        self._loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._target, logits=self._logit))

        # accuracy of the model
        matches = tf.nn.in_top_k(predictions=self._output, targets=tf.argmax(self._target, 1), k=5)
        self._accuracy_top5 = tf.reduce_mean(tf.cast(matches, tf.float32))

        matches = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._logit, 1))
        self._accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    # =========================================================================
    # add regulizers to the loss function
    def _add_l1regulizer(self, w):
        num_layers = len(self._nn_weights)

        if type(w) is float:
            w = [w] * num_layers

        assert len(w) == num_layers, 'Not enough weights for the regularizer.'

        l1_loss = tf.add_n([(s * tf.norm(v, ord=1)) for (v, s) in zip(self._nn_weights, w)])
        self._loss += l1_loss

    def _add_l2regulizer(self, w):
        num_layers = len(self._nn_weights)

        if type(w) is float:
            w = [w] * num_layers

        assert len(w) == num_layers, 'Not enough weights for the regularizer.'

        l2_loss = tf.add_n([(s * tf.nn.l2_loss(v)) for (v, s) in zip(self._nn_weights, w)])
        self._loss += l2_loss

    # =========================================================================
    # define optimizer of the neural network
    def _add_optimizer(self, parameters):
        alg = parameters.get('training_alg', 'GD')
        lr = parameters.get('initial_learning_rate', 0.01)
        dr = parameters.get('decay_rate', 0.95)
        ds = parameters.get('decay_step', 200)

        # define the learning rate
        self._global_counter = tf.Variable(0, dtype=tf.float32, name='global-counter')
        # decayed_learning_rate = learning_rate * dr ^ (train_counter // ds)
        self._learning_rate = tf.train.exponential_decay(lr, self._global_counter, ds, decay_rate=dr, staircase=True)

        # define the appropriate optimizer to use
        if (alg == 0) or (alg == 'GD'):
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
        elif (alg == 1) or (alg == 'RMSProp'):
            self._optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate)
        elif (alg == 2) or (alg == 'Adam'):
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        elif (alg == 3) or (alg == 'AdaGrad'):
            self._optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate)
        elif (alg == 4) or (alg == 'AdaDelta'):
            self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._learning_rate)
        else:
            raise ValueError("Unknown training algorithm.")

        # training and initialization operators
        var_list = self._nn_weights + self._nn_biases
        self._trainOp = self._optimizer.minimize(self._loss, var_list=var_list, global_step=self._global_counter)

        # add backpropagation to the graph
        self._bp_signals = tf.gradients(self._loss, self._bp_signals)

    # =================================================================
    # computing gradients
    def _add_gradient_computations(self):
        # computing gradients
        self._grad_W = tf.gradients(self._loss, self._nn_weights)
        self._grad_b = tf.gradients(self._loss, self._nn_biases)

        # applying gradients to the optimizer
        self._input_gW = tuple([tf.placeholder(dtype=tf.float32, shape=w.get_shape()) for w in self._nn_weights])
        self._input_gb = tuple([tf.placeholder(dtype=tf.float32, shape=b.get_shape()) for b in self._nn_biases])
        gv = [(g, v) for g, v in zip(self._input_gW, self._nn_weights)]
        gv += [(g, v) for g, v in zip(self._input_gb, self._nn_biases)]

        self._apply_gradients = self._optimizer.apply_gradients(gv)

    # =========================================================================
    # add the operations to quantize gradients
    def _add_quantizer(self, parameters):
        pass

    # =========================================================================
    # compute the accuracy of the NN using the given inputs
    def accuracy(self, x, y):
        return self._sess.run(self._accuracy, feed_dict={self._input: x, self._target: y, self._drop_rate: 0.0})

    # =========================================================================
    # One iteration of the training algorithm
    def train(self, x, y, drop_rate=0):
        assert self._trainOp is not None, 'Training algorithm has not been set.'

        self._sess.run(self._trainOp, feed_dict={self._input: x, self._target: y, self._drop_rate: drop_rate})

    # =========================================================================
    # get or set neural network's weights
    def get_weights(self):
        return self._sess.run([self._nn_weights, self._nn_biases])

    def learning_rate(self):
        return self._sess.run(self._learning_rate)

    # =========================================================================
    # Compute the gradients of the parameters of the NN for the given input
    def compute_gradients(self, x, y, drop_rate=0.0):
        assert self._grad_W is not None, 'The operators to compute the gradients have not been defined.'

        return self._sess.run(
            [self._grad_W, self._grad_b], feed_dict={
                self._input: x,
                self._target: y,
                self._drop_rate: drop_rate
            }
        )

    # =========================================================================
    # Apply the gradients externally computed to the optimizer
    def apply_gradients(self, gw, gb):
        assert self._apply_gradients is not None, 'The operators to apply the gradients have not been defined.'

        feed_dict = {self._input_gW: gw, self._input_gb: gb}
        self._sess.run(self._apply_gradients, feed_dict=feed_dict)

    # =========================================================================
    # get forward and backward signals
    # =========================================================================
    # get forward and backward signals
    def get_fw_bp_signals(self, x, target, drop_rate=0):
        if (self._sess is None) or (self._fw_signals is None) or (self._bp_signals is None):
            raise ValueError('The model has not been fully created and initialized.')

        return self._sess.run(
            [self._fw_signals, self._bp_signals],
            feed_dict={
                self._input: x,
                self._target: target,
                self._drop_rate: drop_rate
            }
        )

    # =========================================================================
    # get gradients and signals
    def get_gradients_signals(self, x, target, drop_rate=0):
        if (self._sess is None) or (self._grad_W is None) or (self._fw_signals is None) or (self._bp_signals is None):
            raise ValueError('The model has not been fully created and initialized.')

        return self._sess.run(
            [self._grad_W, self._grad_b, self._fw_signals, self._bp_signals],
            feed_dict={
                self._input: x,
                self._target: target,
                self._drop_rate: drop_rate
            }
        )
