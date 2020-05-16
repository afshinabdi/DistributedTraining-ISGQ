"""
   Fully connected neural network for classification of data, hidden ReLU and final Softmax layers.
   The default network creates a 4 layers fully connected network, (784-300-100-10), for classification of MNSIT data.
"""

import tensorflow as tf
import numpy as np
import scipy.stats as st


class FCModel:
    def __init__(self):
        self._graph = None
        self._sess = None
        self._initializer = None
        self._accuracy = None

        self._optimizer = None
        self._trainOp = None
        self._learning_rate = 0.01
        self._loss = None

        # parameters of the neural network
        self._drop_rate = None
        self._input = None
        self._output = None
        self._logit = None
        self._target = None

        self._nn_weights = []
        self._nn_biases = []
        
        self._num_layers = 0

        # variables to set parameters during training
        self._assign_op = None
        self._input_weights = None
        self._input_biases = None

        # gradients of the neural network
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
    def create_network(self, initial_weights=None, initial_biases=None, layer_shapes=[784, 1000, 300, 100, 10]):
        if initial_weights is None:
            self._create_random_network(layer_shapes)
        else:
            self._create_initialized_network(initial_weights, initial_biases)

    # create neural network with random initial parameters
    def _create_random_network(self, layer_shapes):
        self._num_layers = len(layer_shapes) - 1

        initial_weights = [0] * self._num_layers
        initial_biases = [0] * self._num_layers
        # create initial parameters for the network
        for n in range(self._num_layers):
            initial_weights[n] = st.truncnorm(-2, 2, loc=0,
                                              scale=0.1).rvs((layer_shapes[n], layer_shapes[n + 1]))
            initial_biases[n] = np.ones(layer_shapes[n + 1]) * 0.1

        self._create_initialized_network(initial_weights, initial_biases)

    # create a fully connected neural network with given initial parameters
    def _create_initialized_network(self, initial_weights, initial_biases):
        self._num_layers = len(initial_weights)

        self._fw_signals = []
        self._bp_signals = []
        self._nn_weights = []
        self._nn_biases = []

        input_len = initial_weights[0].shape[0]
        output_len = initial_weights[-1].shape[1]
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._input = tf.placeholder(tf.float32, shape=[None, input_len])
            self._target = tf.placeholder(tf.float32, shape=[None, output_len])
            self._drop_rate = tf.placeholder(tf.float32)

            # create weights and biases of the neural network
            for init_w, init_b in zip(initial_weights, initial_biases):
                w = tf.Variable(init_w.astype(np.float32), dtype=tf.float32)
                b = tf.Variable(init_b.astype(np.float32), dtype=tf.float32)
                self._nn_weights += [w]
                self._nn_biases += [b]

            z = self._input
            for n in range(self._num_layers):
                # add a fully connected layer, relu (for hidden layers) or softmax (for output layer)
                x = tf.nn.dropout(z, rate=self._drop_rate)
                y = tf.matmul(x, self._nn_weights[n]) + self._nn_biases[n]

                self._fw_signals += [x]
                self._bp_signals += [y]

                if n == self._num_layers - 1:
                    z = tf.nn.softmax(y)
                else:
                    z = tf.nn.relu(y)

            # output of the neural network
            self._logit = y
            self._output = z

            # loss function
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._target, logits=self._logit))

            # accuracy of the model
            matches = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._logit, 1))
            self._accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    # =========================================================================
    # add regulizer to the loss function
    def add_regulizer(self, l1_weight=None, l2_weight=None):
        if self._loss is None:
            raise ValueError(
                'The network has not been defined yet. First build the network, then add the regularizer.')

        if l1_weight is None:
            l1_weight = 0.0

        if l2_weight is None:
            l2_weight = 0.0

        num_parameters = len(self._nn_weights)
        if type(l1_weight) is float:
            l1_weight = [l1_weight] * num_parameters

        if type(l2_weight) is float:
            l2_weight = [l2_weight] * num_parameters

        if len(l1_weight) != num_parameters or len(l2_weight) != num_parameters:
            raise ValueError(
                'Number of weights for the l1/l2 regularization is not the same as the number of weights.')

        with self._graph.as_default():
            l1_loss = tf.add_n([(s * tf.norm(w, ord=1))
                                for (w, s) in zip(self._nn_weights, l1_weight)])
            l2_loss = tf.add_n([(s * tf.nn.l2_loss(w))
                                for (w, s) in zip(self._nn_weights, l2_weight)])

            self._loss += (l1_loss + l2_loss)

    # =================================================================
    # update (assign) operator for the parameters of the NN model
    def add_assign_operators(self):
        self._assign_op = []
        self._input_weights = ()
        self._input_biases = ()

        with self._graph.as_default():
            for w in self._nn_weights:
                w_placeholder = tf.placeholder(dtype=tf.float32, shape=w.get_shape())
                w_assign_op = w.assign(w_placeholder)
                self._assign_op.append(w_assign_op)
                self._input_weights += (w_placeholder,)

            for b in self._nn_biases:
                b_placeholder = tf.placeholder(dtype=tf.float32, shape=b.get_shape())
                b_assign_op = b.assign(b_placeholder)
                self._assign_op.append(b_assign_op)
                self._input_biases += (b_placeholder,)

    # =========================================================================
    # define optimizer of the neural network
    def create_optimizer(self, training_algorithm='Adam', learning_rate=0.01, decay_rate=0.95, decay_step=100):
        with self._graph.as_default():
            # define the learning rate
            train_counter = tf.Variable(0, dtype=tf.float32)
            # decayed_learning_rate = learning_rate * decay_rate ^ (train_counter // decay_step)
            self._learning_rate = tf.train.exponential_decay(learning_rate, train_counter, decay_step,
                                                             decay_rate=decay_rate, staircase=True)

            # define the appropriate optimizer to use
            if (training_algorithm == 0) or (training_algorithm == 'GD'):
                self._optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self._learning_rate)
            elif (training_algorithm == 1) or (training_algorithm == 'RMSProp'):
                self._optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 2) or (training_algorithm == 'Adam'):
                self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 3) or (training_algorithm == 'AdaGrad'):
                self._optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 4) or (training_algorithm == 'AdaDelta'):
                self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._learning_rate)
            else:
                raise ValueError("Unknown training algorithm.")

            # =================================================================
            # training operator
            var_list = self._nn_weights + self._nn_biases
            self._trainOp = self._optimizer.minimize(
                self._loss, var_list=var_list, global_step=train_counter)

            # =================================================================
            # computing gradients
            self._grad_W = tf.gradients(self._loss, self._nn_weights)
            self._grad_b = tf.gradients(self._loss, self._nn_biases)

            # applying gradients to the optimizer
            self._input_gW = tuple(
                [tf.placeholder(dtype=tf.float32, shape=w.get_shape()) for w in self._nn_weights])
            self._input_gb = tuple(
                [tf.placeholder(dtype=tf.float32, shape=b.get_shape()) for b in self._nn_biases])
            gv = [(g, v) for g, v in zip(self._input_gW, self._nn_weights)]
            gv += [(g, v) for g, v in zip(self._input_gb, self._nn_biases)]

            self._apply_gradients = self._optimizer.apply_gradients(gv, global_step=train_counter)

            # add backpropagation to the graph
            self._bp_signals = tf.gradients(self._loss, self._bp_signals)

    # =========================================================================
    # initialize the computation graph, if necessary create the initializer and the session

    def initialize(self, device=None):
        if self._initializer is None:
            with self._graph.as_default():
                self._initializer = tf.global_variables_initializer()

            if device is None:
                self._sess = tf.Session(graph=self._graph)
            else:
                config = tf.ConfigProto(log_device_placement=True, device_count=device)
                self._sess = tf.Session(config=config)

        self._sess.run(self._initializer)

    # =========================================================================
    # compute the accuracy of the NN using the given inputs
    def compute_accuracy(self, x, target):
        return self._sess.run(self._accuracy, feed_dict={self._input: x, self._target: target, self._drop_rate: 0})

    # =========================================================================
    # One iteration of the training algorithm with input data
    def train(self, x, y, drop_rate=0):
        if self._trainOp is None:
            raise ValueError('Training algorithm has not been set.')

        self._sess.run(self._trainOp, feed_dict={
                       self._input: x, self._target: y, self._drop_rate: drop_rate})

    # =========================================================================
    # get or set neural network's weights
    def set_weights(self, new_weights, new_biases):
        if self._sess is None or self._assign_op is None:
            raise ValueError('The assign operators has been added to the graph.')

        self._sess.run(self._assign_op, feed_dict={
                       self._input_weights: new_weights, self._input_biases: new_biases})

    def get_weights(self):
        return self._sess.run([self._nn_weights, self._nn_biases])

    def learning_rate(self):
        return self._sess.run(self._learning_rate)

    # =========================================================================
    # Compute the gradients of the parameters of the NN for the given input
    def compute_gradients(self, x, target, drop_rate=0):
        if self._grad_W is None:
            raise ValueError('The operators to compute the gradients have not been defined.')

        return self._sess.run([self._grad_W, self._grad_b],
                              feed_dict={self._input: x, self._target: target, self._drop_rate: drop_rate})

    # =========================================================================
    # Apply the gradients externally computed to the optimizer
    def apply_gradients(self, gw, gb):
        if self._apply_gradients is None:
            raise ValueError('The operators to apply the gradients have not been defined.')

        feed_dict = {self._input_gW: gw, self._input_gb: gb}
        self._sess.run(self._apply_gradients, feed_dict=feed_dict)

    # =========================================================================
    # get forward and backward signals
    def get_fw_bp_signals(self, x, target, drop_rate=0):
        if (self._sess is None) or (self._fw_signals is None) or (self._bp_signals is None):
            raise ValueError('The model has not been fully created and initialized.')

        return self._sess.run([self._fw_signals, self._bp_signals],
                              feed_dict={self._input: x, self._target: target, self._drop_rate: drop_rate})

    # =========================================================================
    # get gradients and signals
    def get_gradients_signals(self, x, target, drop_rate=0):
        if (self._sess is None) or (self._grad_W is None) or (self._fw_signals is None) or (self._bp_signals is None):
            raise ValueError('The model has not been fully created and initialized.')

        return self._sess.run([self._grad_W, self._grad_b, self._fw_signals, self._bp_signals],
                              feed_dict={self._input: x, self._target: target, self._drop_rate: drop_rate})

    # =========================================================================
    # get number of layers
    def get_number_layers(self):
        if (self._sess is None) or (self._grad_W is None):
            raise ValueError('The model has not been fully created and initialized.')

        return len(self._nn_weights)
