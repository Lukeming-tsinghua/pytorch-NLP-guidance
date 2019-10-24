""" Euclidean Loss Layer """

import numpy as np

class EuclideanLossLayer():
    def __init__(self):
        self.accu = 0.
        self.loss = 0.

    def forward(self, logit, gt):
        """
        Inputs: (minibatch)
        - logit: forward results from the last FCLayer, shape(batch_size, 10)
        - gt: the ground truth label, shape(batch_size, 1)
        """
        # TODO: Put your code here
        #  Calculate the average accuracy and loss over the minibatch, and
        #  store in self.accu and self.loss respectively.
        #  Only return the self.loss, self.accu will be used in solver.py.
        # ############################################################################
        batch_size, K = np.shape(logit)
        self.logit = logit
        self.gt = gt

        self.acc = np.sum(np.argmax(logit, axis=1) == np.argmax(gt, axis=1))/batch_size
        self.loss = 1/2/batch_size*np.sum(np.sum(np.square(self.logit-self.gt)))
        ############################################################################
        return self.loss


    def backward(self):
    # Calculate and return the gradient (have the same shape as logit)

    ###########################################################################
         return self.logit-self.gt
    ############################################################################
