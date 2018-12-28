#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-28 上午10:54
# @Author  : 林利芳
# @File    : rnn.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops, clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell, LSTMStateTuple

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class ForgetLSTMCell(LayerRNNCell):
	"""Basic LSTM recurrent network cell.

	The implementation is based on: http://arxiv.org/abs/1409.2329.

	We add forget_bias (default: 1) to the biases of the forget gate in order to
	reduce the scale of forgetting in the beginning of the training.

	It does not allow cell clipping, a projection layer, and does not
	use peep-hole connections: it is the basic baseline.

	For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
	that follows.
	"""
	
	def __init__(self, num_units, forget_bias=1.0,
				 state_is_tuple=True, activation=None, reuse=None, name=None):
		"""Initialize the basic LSTM cell.

		Args:
		  num_units: int, The number of units in the LSTM cell.
		  forget_bias: float, The bias added to forget gates (see above).
			Must set to `0.0` manually when restoring from CudnnLSTM-trained
			checkpoints.
		  state_is_tuple: If True, accepted and returned states are 2-tuples of
			the `c_state` and `m_state`.  If False, they are concatenated
			along the column axis.  The latter behavior will soon be deprecated.
		  activation: Activation function of the inner states.  Default: `tanh`.
		  reuse: (optional) Python boolean describing whether to reuse variables
			in an existing scope.  If not `True`, and the existing scope already has
			the given variables, an error is raised.
		  name: String, the name of the layer. Layers with the same name will
			share weights, but to avoid mistakes we require reuse=True in such
			cases.

		  When restoring from CudnnLSTM-trained checkpoints, must use
		  `CudnnCompatibleLSTMCell` instead.
		"""
		super(ForgetLSTMCell, self).__init__(_reuse=reuse, name=name)
		if not state_is_tuple:
			logging.warn("%s: Using a concatenated state is slower and will soon be "
						 "deprecated.  Use state_is_tuple=True.", self)
		
		# Inputs must be 2-dimensional.
		self.input_spec = base_layer.InputSpec(ndim=2)
		
		self._num_units = num_units
		self._forget_bias = forget_bias
		self._state_is_tuple = state_is_tuple
		self._activation = activation or math_ops.tanh
	
	@property
	def state_size(self):
		return (LSTMStateTuple(self._num_units, self._num_units)
				if self._state_is_tuple else 2 * self._num_units)
	
	@property
	def output_size(self):
		return self._num_units
	
	def build(self, inputs_shape):
		if inputs_shape[1].value is None:
			raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
							 % inputs_shape)
		
		input_depth = inputs_shape[1].value
		h_depth = self._num_units
		self._kernel = self.add_variable(
			_WEIGHTS_VARIABLE_NAME,
			shape=[input_depth + h_depth, 2 * self._num_units])
		self._bias = self.add_variable(
			_BIAS_VARIABLE_NAME,
			shape=[2 * self._num_units],
			initializer=init_ops.zeros_initializer(dtype=self.dtype))
		
		self.built = True
	
	def call(self, inputs, state):
		"""Long short-term memory cell (LSTM).

		Args:
		  inputs: `2-D` tensor with shape `[batch_size, input_size]`.
		  state: An `LSTMStateTuple` of state tensors, each shaped
			`[batch_size, self.state_size]`, if `state_is_tuple` has been set to
			`True`.  Otherwise, a `Tensor` shaped
			`[batch_size, 2 * self.state_size]`.

		Returns:
		  A pair containing the new hidden state, and the new state (either a
			`LSTMStateTuple` or a concatenated state, depending on
			`state_is_tuple`).
		"""
		sigmoid = math_ops.sigmoid
		one = constant_op.constant(1, dtype=dtypes.int32)
		# Parameters of gates are concatenated into one multiply for efficiency.
		if self._state_is_tuple:
			c, h = state
		else:
			c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)
		
		gate_inputs = math_ops.matmul(
			array_ops.concat([inputs, h], 1), self._kernel)
		gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
		
		# i = input_gate, j = new_input, f = forget_gate, o = output_gate
		j, f = array_ops.split(
			value=gate_inputs, num_or_size_splits=2, axis=one)
		i = 1 - f
		forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
		# Note that using `add` and `multiply` instead of `+` and `*` gives a
		# performance improvement. So using those at the cost of readability.
		add = math_ops.add
		multiply = math_ops.multiply
		new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))), multiply(sigmoid(i), self._activation(j)))
		new_h = new_c
		
		if self._state_is_tuple:
			new_state = LSTMStateTuple(new_c, new_h)
		else:
			new_state = array_ops.concat([new_c, new_h], 1)
		return new_h, new_state


class IndRNNCell(LayerRNNCell):  # 继承 LayerRNNCell
	
	def __init__(self,
				 num_units,
				 recurrent_min_abs=0,
				 recurrent_max_abs=None,
				 recurrent_kernel_initializer=None,
				 input_kernel_initializer=None,
				 activation=None,
				 reuse=None,
				 name=None):
		super(IndRNNCell, self).__init__(_reuse=reuse, name=name)
		
		self.input_spec = base_layer.InputSpec(ndim=2)
		
		# initialization
		self._num_units = num_units
		self._recurrent_min_abs = recurrent_min_abs
		
		self._recurrent_max_abs = recurrent_max_abs
		self._recurrent_recurrent_kernel_initializer = recurrent_kernel_initializer
		self._input_kernel_initializer = input_kernel_initializer
		self._activation = activation or nn_ops.relu
	
	@property
	def state_size(self):
		return self._num_units
	
	@property
	def output_size(self):
		return self._num_units
	
	def build(self, inputs_shape):
		'''construct the IndRNN Cell'''
		if inputs_shape[1].value is None:
			raise ValueError("Expected input shape[1] is known")
		
		input_depth = inputs_shape[1]
		if self._input_kernel_initializer is None:
			self._input_kernel_initializer = init_ops.random_normal_initializer(mean=0,
																				stddev=1e-3)
		# matrix W
		self._input_kernel = self.add_variable(
			"input_kernel",
			shape=[input_depth, self._num_units],
			initializer=self._input_kernel_initializer
		)
		
		if self._recurrent_recurrent_kernel_initializer is None:
			self._recurrent_recurrent_kernel_initializer = init_ops.constant_initializer(1.)
		
		# matrix U
		self._recurrent_kernel = self.add_variable(
			"recurrent_kernel",
			shape=[self._num_units],
			initializer=self._recurrent_recurrent_kernel_initializer
		)
		
		# Clip the U to min - max
		if self._recurrent_min_abs:
			abs_kernel = math_ops.abs(self._recurrent_kernel)
			min_abs_kernel = math_ops.maximum(abs_kernel, self._recurrent_min_abs)
			self._recurrent_kernel = math_ops.multiply(
				math_ops.sign(self._recurrent_kernel),
				min_abs_kernel
			)
		if self._recurrent_max_abs:
			self._recurrent_kernel = clip_ops.clip_by_value(
				self._recurrent_kernel,
				-self._recurrent_max_abs,
				self._recurrent_max_abs
			)
		
		self._bias = self.add_variable(
			"bias",
			shape=[self._num_units],
			initializer=init_ops.zeros_initializer(dtype=self.dtype)
		)
		# built finished
		self.built = True
	
	def call(self, inputs, state):
		'''output = new state = activation(W * x + U (*) h_t-1 + b)'''
		
		gate_inputs = math_ops.matmul(inputs, self._input_kernel)
		# (*)
		state_update = math_ops.multiply(state, self._recurrent_kernel)
		gate_inputs = math_ops.add(gate_inputs, state_update)
		gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
		output = self._activation(gate_inputs)
		return output, output
