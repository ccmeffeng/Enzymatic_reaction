import tensorflow as tf


def rnn_dense(name, x_len, num_units, inputs, state, bias=0):
    with tf.variable_scope(name, reuse=None):
        U = tf.get_variable("U", shape=[num_units, num_units], dtype=tf.float32)
        W = tf.get_variable("W", shape=[x_len, num_units], dtype=tf.float32)
        output = tf.matmul(inputs, W) + tf.matmul(state, U)
        if bias == 0:
            b = tf.get_variable('b', shape=[num_units], dtype=tf.float32)
            output += b
        if bias == 1:
            b = tf.get_variable('b', shape=(num_units,), dtype=tf.float32, initializer=tf.ones_initializer())
            output += b
    return output


class BasicRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):
            # todo: implement the new_state calculation given inputs and state
            x_size = inputs.get_shape().as_list()[1]
            new_state = self._activation(rnn_dense('rnn', x_size, self.state_size, inputs, state))
        return new_state, new_state


class GRUCell(tf.contrib.rnn.RNNCell):
    '''Gated Recurrent Unit cell (http://arxiv.org/abs/1406.1078).'''

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "gru_cell", reuse=self._reuse):
            # We start with bias of 1.0 to not reset and not update.
            # todo: implement the new_h calculation given inputs and state
            x_size = inputs.get_shape().as_list()[1]
            # update gate z:
            z = tf.nn.sigmoid(rnn_dense('z', x_size, self.state_size, inputs, state, bias=1))
            # reset gate r :
            r = tf.nn.sigmoid(rnn_dense('r', x_size, self.state_size, inputs, state, bias=1))

            Wc = tf.get_variable("Wc", shape=(x_size, self.state_size))
            Uc = tf.get_variable("Uc", shape=(self.state_size, self.state_size))
            _h = tf.nn.tanh(tf.matmul(inputs, Wc) + tf.matmul(tf.multiply(r, state), Uc))
            new_h = tf.multiply(1 - z, state) + tf.multiply(z, _h)

        return new_h, new_h


class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    '''Basic LSTM cell (http://arxiv.org/abs/1409.2329).'''

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units, self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_lstm_cell", reuse=self._reuse):
            c, h = state
            # For forget_gate, we add forget_bias of 1.0 to not forget in order to reduce the scale of forgetting in the beginning of the training.
            # todo: implement the new_c, new_h calculation given inputs and state (c, h)
            x_size = inputs.get_shape().as_list()[1]
            i = tf.nn.sigmoid(rnn_dense('i', x_size, self.output_size, inputs, h, bias=1))
            o = tf.nn.sigmoid(rnn_dense('o', x_size, self.output_size, inputs, h))
            f = tf.nn.sigmoid(rnn_dense('f', x_size, self.output_size, inputs, h))
            _C = tf.nn.tanh(rnn_dense('C', x_size, self.output_size, inputs, h))
            new_c = tf.multiply(f, c) + tf.multiply(i, _C)
            new_h = tf.multiply(o, tf.nn.tanh(new_c))

            return new_h, (new_c, new_h)
