import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  num_train  = X.shape[1]
  num_classes = W.shape[0]

  fs = W.dot(X)
  for i in xrange(num_train):
    f = fs[:, i]
    # shift the f value to achieve numerial stability
    f -= np.max(f)

    # probability interpretation for each class
    p = np.exp(f) / np.sum(np.exp(f), axis=0)

    # p = np.exp(f[y[i]]) / np.sum( f, axis=0 )
    loss += - f[y[i]] + np.log( np.sum( np.exp(f), axis=0) )

    for j in xrange(num_classes):
      dW[j,:] +=  p[j] * X[:,i]

    dW[y[i],:] -= X[:, i]
    ##################################################
    # So the gradient in total is:
    # dW = (p[j] - 1) * X[:, i]     , if j == i
    #    =  p[j]      * X[:, j]     , otherwise
    #
    # just use chain rule to calculate gradient
    ##################################################

  # calculate average batch gradient
  dW /= num_train
  # calculate average batch loss
  loss /= num_train
  # regularization
  loss += 0.5 * reg * np.sum( W ** 2)

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
  num_train  = X.shape[1]
  num_classes = W.shape[0]  

  f = W.dot(X)
  f -= np.amax(f, axis=0)

  p = np.exp(f) / np.sum(np.exp(f), axis=0)
  
  loss = np.sum(- f[y, range(num_train)] + np.log( np.sum( np.exp(f), axis=0) ))

  # the universal component gradient
  dW += p.dot(X.T)

  # the correct specified component gradient 
  select_correct = np.zeros_like(f)
  select_correct[y, range(num_train)] = 1
  dW -= select_correct.dot(X.T)

  loss /= num_train
  dW /= num_train

  return loss, dW
