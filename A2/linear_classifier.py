"""
Implements linear classifiers with SVM and Softmax losses in PyTorch.
WARNING: you SHOULD NOT use `.to()` or `.cuda()` in each implementation block.
"""
import random
from abc import abstractmethod
from typing import Callable, Dict, List, Optional

import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from linear_classifier.py!")


# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier:
    """An abstract class for the linear classifiers"""

    # Note: We will re-use `LinearClassifier' in both SVM and Softmax
    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A tensor of shape (D, C) containing weights of the model.
        - X_batch: A tensor of shape (N, D) containing a minibatch of N data
          points, where each data point has the dimension of D.
        - y_batch: A tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the multiclass SVM loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)


# ************************************************ #
#                  Section 1: SVM                  #
# ************************************************ #


def svm_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples. When you implement the regularization over W, please DO NOT
    multiply the regularization term by 1/2 (no coefficient).

    Inputs:
    - W: A tensor of shape (D, C) containing weights.
    - X: A tensor of shape (N, D) containing a minibatch of data.
    - y: A tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = W.t().mv(X[i])
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                ##############################################################
                # TODO: Compute the gradient of the SVM term of the loss     #
                # function and store it on dW. (part 1) Rather than first    #
                # computing the loss and then computing the derivative, it   #
                # is simple to compute the derivative at the same time that  #
                # the loss is being computed.                                #
                ##############################################################
                # Replace "pass" with your code (do not modify this line)
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]
                ##############################################################
                #                      END OF YOUR CODE                      #
                ##############################################################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * torch.sum(W * W)

    ##########################################################################
    # TODO: Compute the gradient of the loss function w.r.t. the             #
    # regularization term and add it to dW. (part 2) Don't forget to divide  #
    # your gradient by num_train first, if you have not done so in part 1.   #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    dW /= num_train
    dW += reg * 2 * W 
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    return loss, dW


def svm_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, vectorized implementation. When you implement
    the regularization over W, please DO NOT multiply the regularization term
    by 1/2 (no coefficient). The inputs and outputs are the same as
    svm_loss_naive.

    Inputs:
    - W: A tensor of shape (D, C) containing weights.
    - X: A tensor of shape (N, D) containing a minibatch of data.
    - y: A tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    ##########################################################################
    # TODO: Implement a vectorized version of the structured SVM loss,       #
    # storing the result in loss.                                            #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X @ W
    correct_class_score = scores[torch.arange(num_train), y]
    correct_class_score = correct_class_score.view(num_train, -1)
    margins = (scores - correct_class_score + 1)
    margins[margins < 0] = 0
    margins[torch.arange(num_train), y] = 0

    loss += margins[margins > 0].sum()
    loss /= num_train
    loss += reg * torch.sum(W * W)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    ##########################################################################
    # TODO: Implement a vectorized version of the gradient for the           #
    # structured SVM loss, storing the result in dW.                         #
    #                                                                        #
    # Hint: Instead of computing the gradient from scratch, it may be easier #
    # to reuse some of the intermediate values that you used to compute the  #
    # loss.                                                                  #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    mask = torch.zeros(margins.shape, device=margins.device, dtype=margins.dtype)
    mask[margins > 0] = 1
    dW += X.t() @ mask
    counts = -torch.sum(mask, axis=1)
    mask = torch.zeros(margins.shape, device=margins.device, dtype=margins.dtype)
    mask[torch.arange(num_train), y] = counts
    dW += X.t() @ mask 

    dW /= num_train
    dW += reg * 2 * W 
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    return loss, dW


def sample_batch(
    X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """
    X_batch = None
    y_batch = None
    ##########################################################################
    # TODO: Store the data in X_batch and their corresponding labels in      #
    # y_batch; after sampling, X_batch should have shape (batch_size, dim)   #
    # and y_batch should have shape (batch_size,)                            #
    #                                                                        #
    # Hint: torch.randint; you may want to borrow the device from X.         #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    idx = torch.randint(0, num_train, (batch_size,))
    X_batch = X[idx]
    y_batch = y[idx]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return X_batch, y_batch


def train_linear_classifier(
    loss_func: Callable,
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-3,
    reg: float = 1e-5,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
      and reg as input, and output a tuple of (loss, dW)
    - W: A tensor of shape (D, C) giving the initial weights of the
      classifier. If W is None then it will be initialized here.
    - X: A tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A tensor of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at
      each training iteration.
    """
    # assume y takes values 0...K-1 where K is number of classes
    num_train, dim = X.shape
    if W is None:
        # lazily initialize W
        num_classes = torch.max(y) + 1
        W = 0.000001 * torch.randn(
            dim, num_classes, device=X.device, dtype=X.dtype
        )
    else:
        num_classes = W.shape[1]

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        # sample_batch function is implemented above
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # evaluate loss and gradient
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

        # perform parameter update
        ######################################################################
        # TODO: Update the weights using the gradient and the learning rate. #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        W -= learning_rate * grad 
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        if verbose and it % 100 == 0:
            print("iteration %4d / %4d: loss %f" % (it, num_iters, loss))

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - W: A tensor of shape (D, C), containing weights of a model
    - X: A tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: int64 tensor of shape (N,) giving predicted labels for each
      element of X. Each element of y_pred should be between 0 and C - 1.
    """
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
    ##########################################################################
    # TODO: Implement this method, and store the predicted labels in y_pred. #
    # Hint: torch.argmax                                                     #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    y_pred = torch.argmax(X @ W, axis=1)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y_pred


def svm_get_search_params():
    """
    Return candidate hyperparameters for the SVM model. You should provide
    at least two params for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates,
      e.g., [1e0, 1e1, ...]
    """

    learning_rates = []
    regularization_strengths = []

    ##########################################################################
    # TODO: add your own hyper parameter lists.                              #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    learning_rates.append(1e-3)
    learning_rates.append(1e-2)
    learning_rates.append(1e-1)
    regularization_strengths.append(1e0)
    regularization_strengths.append(1e1)
    regularization_strengths.append(1e-1)
    regularization_strengths.append(3e-1)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    return learning_rates, regularization_strengths


def test_one_param_set(
    cls: LinearClassifier,
    data_dict: Dict[str, torch.Tensor],
    lr: float,
    reg: float,
    num_iters: int = 2000,
):
    """
    Train a single LinearClassifier instance and return the learned instance
    with train/val accuracy.

    Inputs:
    - cls (LinearClassifier): a newly-created LinearClassifier instance.
      train/val should be performed over this instance.
    - data_dict (dict): a dictionary that includes ['X_train', 'y_train',
      'X_val', 'y_val'] as the keys for training a classifier.
    - lr (float): learning rate parameter for training a SVM instance.
    - reg (float): a regularization weight for training a SVM instance.
    - num_iters (int, optional): a number of iterations to train

    Returns:
    - cls (LinearClassifier): a trained LinearClassifier instances with
      (['X_train', 'y_train'], lr, reg) for num_iter times.
    - train_acc (float): training accuracy of the svm_model
    - val_acc (float): validation accuracy of the svm_model
    """
    train_acc = 0.0  # The accuracy is simply the fraction of data points
    val_acc = 0.0  # that are correctly classified.
    ###########################################################################
    # TODO: Write code that, train a linear SVM on the training set, compute  #
    # its accuracy on the training and validation sets.                       #
    #                                                                         #
    # Hint: Once you are confident that your validation code works, you       #
    # should rerun the validation code with the final value for num_iters.    #
    # Before that, please test with small num_iters first.                    #
    # Hint 2: You might want to use .item() to get float from a tensor when   #
    # computing train_acc and val_acc.                                        #
    ###########################################################################
    # Feel free to uncomment this line at the very beginning for debugging,
    # but don't forget to remove this before submitting your final version
    # num_iters = 100

    # Replace "pass" with your code (do not modify this line)
    loss_history = cls.train(data_dict['X_train'], data_dict['y_train'], lr, reg, num_iters,)

    y_pred_train = cls.predict(data_dict['X_train'])
    y_pred_val = cls.predict(data_dict['X_val'])

    train_acc = 100 * torch.mean( (y_pred_train==data_dict['y_train']).float() ).item()
    val_acc = 100 * torch.mean( (y_pred_val==data_dict['y_val']).float() ).item()
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    return cls, train_acc, val_acc


# ************************************************ #
#                Section 2: Softmax                #
# ************************************************ #


def softmax_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, naive implementation (with loops).
    When you implement the regularization over W, please DO NOT multiply the
    regularization term by 1/2 (no coefficient).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A tensor of shape (D, C) containing weights.
    - X: A tensor of shape (N, D) containing a minibatch of data.
    - y: A tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an tensor of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful  #
    # here, it is easy to run into numeric instability;                      #
    # Check "Practical issues: numeric stability" in the ipynb.              #
    # Plus, don't forget the regularization!                                 #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    num_classes = W.shape[1]
    num_train = X.shape[0]

    C = 0
    for i in range(num_train):
        for j in range(num_classes):
            if W.t()[j] @ X[i] > C:
                C = -(W.t()[j] @ X[i])
    
    for i in range(num_train):

        scores = X[i] @ W
        y_score = scores[y[i]]

        loss -= torch.log(torch.exp(y_score) / torch.sum(torch.exp(scores)))

        dW[:, y[i]] -= X[i]
        dW += (X[i].view(-1, 1) @ torch.exp(scores).view(1, -1)) / torch.sum(torch.exp(scores))
    
    loss /= num_train
    dW /= num_train

    loss += reg * torch.sum(W * W)
    dW += reg * 2 * W 
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    return loss, dW


def softmax_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, vectorized version.
    When you implement the regularization over W, please DO NOT multiply the
    regularization term by 1/2 (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient without explicit loops.#
    # Store the loss in loss and the gradient in dW. If you are not careful  #
    # here, it is easy to run into numeric instability;                      #
    # Check "Practical issues: numeric stability" in the ipynb.              #
    # Plus, don't forget the regularization!                                 #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    num_train = X.shape[0]
    num_class = W.shape[1]

    C = 0
    for i in range(num_train):
        for j in range(num_class):
            if W.t()[j] @ X[i] > C:
                C = -(W.t()[j] @ X[i])
    
    scores = X @ W
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[torch.arange(num_train), y] = 1

    sum_scores = torch.log(torch.sum(torch.exp(scores + C), axis=1))

    loss -= torch.sum(scores[mask])
    loss += torch.sum(sum_scores)

    loss /= num_train
    loss += reg * torch.sum(W * W)

    dL_dWj = torch.exp(scores) / torch.sum(torch.exp(scores), axis=1).view(-1, 1)
    dW += X.t() @ dL_dWj
    dW -= X.t() @ mask.double()
    dW /= num_train
    dW += 2 * reg * W
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    return loss, dW


def softmax_get_search_params():
    """
    Return candidate hyperparameters for the Softmax model. You should provide
    at least two params for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """
    learning_rates = []
    regularization_strengths = []

    ##########################################################################
    # TODO: Add your own hyper parameter lists. This might be similar to the #
    # hyperparameters that you used for the SVM, but you may need to select  #
    # different hyperparameters to achieve good performance with the softmax #
    # classifier.                                                            #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    learning_rates.append(0.01)
    learning_rates.append(1e-2)
    learning_rates.append(1e-1)
    regularization_strengths.append(1e-1)
    regularization_strengths.append(1e-2)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    return learning_rates, regularization_strengths

