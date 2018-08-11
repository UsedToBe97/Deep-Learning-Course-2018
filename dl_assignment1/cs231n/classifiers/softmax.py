import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  N, D = X.shape
  D, C = W.shape
  scores = X.dot(W)
  ex = np.zeros((N, C))
  o = ex
  for i in range(N):
    for j in range(C):
      ex[i][j] = np.exp(scores[i][j])
    tot = 0
    for j in range(C):
      tot += ex[i][j]
    for j in range(C):
      o[i][j] = ex[i][j] / tot
  #ex = np.exp(scores) # N * C
  #o = ex / ex.sum(axis=1, keepdims=True) # N * C
  loss = reg * np.sum(W * W) * 0.5
  for i in range(N):
    #print np.log(o[i][y[i]])
    loss -= np.log(o[i][y[i]]) / N

  dW = o
  for i in range(N):
    dW[i][y[i]] -= 1.0
  dW = np.dot(np.transpose(X), dW) / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  N, D = X.shape
  D, C = W.shape
  scores = X.dot(W)
  ex = np.exp(scores) # N * C
  o = ex / ex.sum(axis=1, keepdims=True) # N * C
  loss = reg * np.sum(W * W) * 0.5
  loss -= np.sum(np.log(o[np.arange(N), y])) / N

  dW = o
  dW[np.arange(N), y]-= 1.0
  dW = np.dot(np.transpose(X), dW) / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

