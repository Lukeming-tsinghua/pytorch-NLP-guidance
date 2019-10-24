import numpy as np

def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    N = np.shape(input)[0]

    x = np.matmul(input, W)
    z = 1/(1 + np.exp(-x))
    soft = np.reshape(np.sum(np.exp(z), axis=1), (N, 1))
    p = np.exp(z) / np.repeat(soft, 10, axis=1)
    loss = -(np.sum(np.sum(label * np.log(p))) + lamda*np.sum(np.sum(np.square(W))))/N
    gradient = np.matmul(np.transpose(input),(p-label)*z*(1-z)) + 2*lamda*W
    prediction = np.argmax(p, axis=1)
    ############################################################################

    return loss, gradient, prediction
