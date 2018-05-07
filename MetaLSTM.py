import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl as rnn_cell
from tensorflow.python.ops.math_ops import sigmoid, tanh

class MetaLSTMCell(rnn_cell.RNNCell):

    def __init__(self, 
                num_units, 
                metalstm_cell=None, 
                activation= tanh):
        """Initialize the basic LSTM cell.
        Args:
        num_units: int, The number of units in the LSTM cell.
        metalstm_cell: lstm_cell, the cell used as meta networks.
        activation: Activation function of the inner states.
        state_is_tuple: False
        """
        self._num_units = num_units
        self._activation = activation

        self._meta_cell = metalstm_cell
        self._meta_num_units= self._meta_cell.output_size
        self._total_num_units = self._num_units + self._meta_num_units

    @property
    def state_size(self):
        return 2 * self._total_num_units

    @property
    def output_size(self):
        return self._num_units

    def getMetaResults(self, meta_output, input, dimensions, scope="meta"):
        """calculate the gates results of basic lstm with meta-lstm network"""    
        # with tf.variable_scope('z_trans'):
        #     meta_output = rnn_cell._linear(meta_output, self._meta_num_units, False)

        with tf.variable_scope(scope):
            W_matrix_list = []
            input_shape = int(input.get_shape()[-1])

            #generate parameters of basic lstm
            for i in np.arange(4):
                P = tf.get_variable('P{}'.format(i), shape=[self._meta_num_units, dimensions],
                    initializer=tf.uniform_unit_scaling_initializer(),dtype=tf.float32)
                Q = tf.get_variable('Q{}'.format(i), shape=[self._meta_num_units, input_shape], 
                    initializer=tf.uniform_unit_scaling_initializer(),dtype=tf.float32)
                
                _W_matrix = tf.matmul(tf.reshape(tf.matrix_diag(meta_output),[-1, self._meta_num_units]), P)
                _W_matrix = tf.reshape(_W_matrix, [-1, self._meta_num_units, dimensions])
                _W_matrix = tf.matmul(tf.reshape(tf.transpose(_W_matrix, [0,2,1]), [-1, self._meta_num_units]), Q)
                _W_matrix = tf.reshape(_W_matrix, [-1, dimensions, input_shape])
                W_matrix_list.append(_W_matrix)
            W_matrix = tf.concat(values=W_matrix_list, axis=1)
            Bias = rnn_cell._linear(meta_output, 4*dimensions, False)

            result = tf.matmul(W_matrix, tf.expand_dims(input, -1))
            result = tf.add(tf.reshape(result, [-1, 4*dimensions]), Bias)
            return result


    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM) with meta_lstm_networks"""
        with tf.variable_scope(scope or type(self).__name__):
            # Parameters of gates are concatenated into one multiply for efficiency.
            total_h, total_c = tf.split(axis=1, num_or_size_splits=2, value=state)
            h = total_h[:, 0:self._num_units]
            c = total_c[:, 0:self._num_units]
            meta_state = tf.concat(values=[total_h[:, self._num_units:], total_c[:, self._num_units:]], axis=1)
            meta_input = tf.concat(values=[inputs, h], axis=1)

            #get outputs from meta-lstm
            meta_output, meta_new_state = self._meta_cell(meta_input, meta_state)

            #calculate gates of basic lstm
            input_concat = tf.concat(values=[inputs, h], axis=1)
            lstm_gates= self.getMetaResults(meta_output, input_concat, self._num_units, scope = 'meta_result')
            i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=lstm_gates)
            new_c = (c * sigmoid(f) + sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)

            #update new states
            meta_h, meta_c = tf.split(axis=1, num_or_size_splits=2, value=meta_new_state)
            new_total_h = tf.concat(values=[new_h, meta_h], axis=1)
            new_total_c = tf.concat(values=[new_c, meta_c], axis=1)
            new_total_state = tf.concat(values=[new_total_h, new_total_c], axis=1)

            return  new_h, new_total_state