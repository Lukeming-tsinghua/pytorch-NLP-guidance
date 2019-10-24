""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = np.zeros(1, dtype='f')

    def forward(self, logit, gt):
        """
          Inputs: (minibatch)
          - logit: forward results from the last FCLayer, shape(batch_size, 10)
          - gt: the ground truth label, shape(batch_size, 1)
        """

        ############################################################################
        # TODO: Put your code here
        batch_size,K = np.shape(logit)
        main_term = np.exp(logit)
        norm_term = np.reshape(np.sum(main_term, axis=1), (batch_size,1))
        norm_term = np.repeat(norm_term, K, axis=1)

        self.prob = main_term/norm_term
        self.acc = np.sum(np.argmax(logit, axis=1) == np.argmax(gt, axis=1)) / batch_size
        self.gt = gt
        self.loss = - np.sum(np.sum(self.gt*np.log(self.prob)))/batch_size

        # Calculate the average accuracy and loss over the minibatch, and
        # store in self.accu and self.loss respectively.
        # Only return the self.loss, self.accu will be used in solver.py.
        ############################################################################

        return self.loss


    def backward(self):

    #############################################################################
    #TODO: Put your code here# Calculate and return the gradient (have the same shape as logit)
        return (self.prob-self.gt)
    ############################################################################
