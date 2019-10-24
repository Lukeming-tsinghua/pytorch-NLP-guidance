""" ReLU Layer """

import numpy as np

class ReLULayer():
	def __init__(self):
		"""
		Applies the rectified linear unit function element-wise: relu(x) = max(x, 0)
		"""
		self.trainable = False # no parameters

	def forward(self, Input):

		############################################################################
	    # TODO: Put your code here
		self.input = Input
		self.output = (self.input>0)*self.input
		# Apply ReLU activation function to Input, and return results.

		return self.output
	    ############################################################################


	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here
		delta = delta*(self.input>0)
		# Calculate the gradient using the later layer's gradient: delta
		return delta
	    ############################################################################
