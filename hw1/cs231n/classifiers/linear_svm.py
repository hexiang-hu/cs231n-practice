import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    
    cnt = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # if margin > 0, the loss function is updated, so that gradient is 
        # also updated, and for the linear case, the grad_Wj = X_i
        dW[j,:] += X[:, i]
        # use a cnt variable to record the number of updates to the 
        # correct class parameter
        cnt += 1
    
    dW[y[i],:] -= X[:, i] * cnt

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  # Also use the averaged gradient of full batch
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W ** 2)
  # add regularization gradient
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train = X.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = W.dot(X)

  # select all correct class score
  correct_class_score = scores[y, range(num_train) ]
  
  margins = np.maximum(scores - correct_class_score + 1, 0)
  margins[y, range(num_train)] = 0

  loss = np.sum(margins) / num_train
  loss += 0.5 * reg * np.sum(W ** 2)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  select_wrong = np.zeros(margins.shape)
  select_wrong[margins > 0] = 1

  select_correct = np.zeros(margins.shape)
  select_correct[y, range(num_train)] = np.sum(select_wrong, axis=0)

  dW = select_wrong.dot(X.T)
  dW -= select_correct.dot(X.T)
  dW /= num_train
  
  # add regularization gradient
  dW += reg * W

  return loss, dW
