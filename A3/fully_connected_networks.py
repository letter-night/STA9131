"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
As a practice, you are NOT allowed to use torch.nn ops.
"""
import torch

from a3_helpers import softmax_loss
from common import Solver


def hello():
    """
    This is a sample function that we will try to import and run to ensure
    that our environment is correctly set up on Google Colab.
    """
    print('Hello from fully_connected_networks.py!')


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Compute the forward pass for a linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: Output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        ######################################################################
        # TODO: Implement the linear forward pass. Store the result in `out` #
        # Note that you need to reshape the input into rows.                 #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        out = x.reshape(x.shape[0], -1) @ w.type_as(x) + b.type_as(x) # (N, D) @ (D, M) + (M,) = (N, M)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Compute the backward pass for a linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: A tensor containing input data, of shape (N, d_1, ... d_k)
          - w: A tensor of weights, of shape (D, M)
          - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        ######################################################################
        # TODO: Implement the linear backward pass.                          #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        dout = dout.type_as(x)
        dx = (dout @ w.type_as(x).t()).reshape(x.shape) # (N, M) @ (M, D) = (N, D) -> (N, d_1, ,, d_k)
        dw = x.reshape(x.shape[0], -1).t() @ dout # (D, N) @ (N, M) = (D, M)
        db = dout.sum(axis=0) # sum((N, M), axis=0) = (M,)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Compute the forward pass for a layer of rectified linear unit (ReLU).
        Input:
        - x: A tensor containing input data, of any shape
        Returns a tuple of:
        - out: Output; a tensor of the same shape as x
        - cache: x
        """
        out = None
        ######################################################################
        # TODO: Implement the ReLU forward pass.                             #
        # You should not change the input tensor with an in-place operation. #
        # You are NOT allowed to use torch.relu                              #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        out = torch.maximum(torch.tensor(0), x)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Compute the backward pass for a layer of rectified linear unit (ReLU).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: A tensor containing input data, of the same shape as dout
        Returns:
        - dx: Gradient with respect to x, of the same shape as dout
        """
        dx, x = None, cache
        ######################################################################
        # TODO: Implement the ReLU backward pass.                            #
        # You should not change the input tensor with an in-place operation. #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        dx = dout * (x > 0)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs a linear transform followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output of the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input
    dimension of D, a hidden dimension of H, and perform classification over
    C classes.
    The architecture should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it will
    interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this data type. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg

        ######################################################################
        # TODO: Initialize the weights and biases of the two-layer net.      #
        # Weights should be initialized from a Gaussian centered at 0.0 with #
        # the standard deviation equal to weight_scale, and biases should be #
        # initialized to zero. All weights and biases should be stored in    #
        # the dictionary self.params, with the first layer weights and       #
        # biases using the keys 'W1' and 'b1' and second layer weights and   #
        # biases using the keys 'W2' and 'b2'.                               #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        self.params["W1"] = torch.randn(input_dim, hidden_dim) * weight_scale
        self.params["b1"] = torch.zeros(hidden_dim)
        self.params["W2"] = torch.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = torch.zeros(num_classes)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] is the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        scores = None
        ######################################################################
        # TODO: Implement the forward pass for the two-layer net, computing  #
        # the class scores for X and storing them in the scores variable.    #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        x, cache1 = Linear_ReLU.forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = Linear.forward(x, self.params['W2'], self.params['b2'])
         
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ######################################################################
        # TODO: Implement the backward pass for the two-layer net.           #
        # Store the loss in the loss variable and gradients in the grads     #
        # dictionary. Compute data loss using softmax, and make sure that    #
        # grads[k] holds the gradients for self.params[k]. Don't forget to   #
        # add L2 regularization!                                             #
        # NOTE: To ensure your implementation matches ours and you pass the  #
        # automated tests, make sure that your L2 regularization includes    #
        # a factor of 0.5 to simplify the expression for the gradient.       #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * torch.sum(torch.sum(self.params['W1']**2) + torch.sum(self.params['W2']**2))

        dout, dW, db = Linear.backward(dout, cache2)
        grads['W2'] = dW + self.reg * self.params['W2']
        grads['b2'] = db 
        
        dout, dW, db = Linear_ReLU.backward(dout, cache1)
        grads['W1'] = dW + self.reg * self.params['W1']
        grads['b1'] = db 
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    solver = None
    ##########################################################################
    # TODO: Use a Solver instance to train a TwoLayerNet that achieves at    #
    # least 50% accuracy on the validation set.                              #
    # We will use the default SGD, so do NOT specify update_rule.            #
    # Hint: Experiment with learning_rate in optim_config and lr_decay.      #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    solver = Solver(model, data_dict)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return solver


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden
    layers, ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L-1) - linear - softmax

    where dropout is optional, and the {...} block is repeated (L-1) times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of hidden layers.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving the drop probability for
          networks with dropout. If dropout=0, then the network should not use
          dropout.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - seed: If not None, then pass this random seed to the dropout layers.
          This will make the dropout layers deterministic so we can gradient
          check the model.
        - dtype: A torch data type object; all computations will be performed
          using this data type. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ######################################################################
        # TODO: Initialize the parameters of the network and store all       #
        # values in the self.params dictionary. Store weights and biases for #
        # the first layer in W1 and b1; for the second layer use W2 and b2,  #
        # etc. Weights should be initialized from a normal distribution      #
        # centered at 0 with standard deviation equal to weight_scale.       #
        # Biases should be initialized to zero.                              #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)       
        for l, (i,j) in enumerate(zip([input_dim, *hidden_dims], [*hidden_dims, num_classes])):
            self.params[f"W{l+1}"] = torch.randn(i, j, device='cuda') * weight_scale
            self.params[f"b{l+1}"] = torch.zeros(j, device='cuda')
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        # Note: When using dropout, we need to pass a dropout_param dictionary
        # to each dropout layer, so that the layer knows the dropout
        # probability and the mode (train / test). You can pass the same
        # dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'use_dropout': self.use_dropout,
          'dropout_param': self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dropout_param = checkpoint['dropout_param']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute the loss and gradient for the fully-connected net.
        Input / Output: Same as TwoLayerNet above.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Note: Set train/test mode for dropout param, as its behaves
        # differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        scores = None
        ######################################################################
        # TODO: Implement the forward pass for the fully-connected net,      #
        # computing the class scores for X and storing them in the scores    #
        # variable. When using dropout, you need to pass self.dropout_param  #
        # to each dropout forward pass.                                      #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        cache = {}
        if self.use_dropout:
            dropout_cache = {}
        for i in range(self.num_layers-1):
            X, cache[i] = Linear_ReLU.forward(X, self.params[f"W{i+1}"], self.params[f"b{i+1}"])
            if self.use_dropout:
                X, dropout_cache[i] = Dropout.forward(X, self.dropout_param)
        X, cache[self.num_layers-1] = Linear.forward(
            X, self.params[f"W{self.num_layers}"], self.params[f"b{self.num_layers}"])
        
        scores = X 
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ######################################################################
        # TODO: Implement the backward pass for the fully-connected net.     #
        # Store the loss in the loss variable and gradients in the grads     #
        # dictionary. Compute data loss using softmax, and make sure that    #
        # grads[k] holds the gradients for self.params[k]. Don't forget to   #
        # add L2 regularization!                                             #
        # NOTE: To ensure your implementation matches ours and you pass the  #
        # automated tests, make sure that your L2 regularization includes    #
        # a factor of 0.5 to simplify the expression for the gradient.       #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * torch.tensor([torch.sum(W**2) for k, W in self.params.items() if 'W' in k]).sum()

        dout, dW, db = Linear.backward(dout, cache[self.num_layers-1])

        grads[f'W{self.num_layers}'] = dW + self.reg * self.params[f'W{self.num_layers}']
        grads[f"b{self.num_layers}"] = db 

        for i in reversed(range(self.num_layers-1)):
            if self.use_dropout:
                dout = Dropout.backward(dout, dropout_cache[i])
            dout, dW, db = Linear_ReLU.backward(dout, cache[i])

            grads[f"W{i+1}"] = dW + self.reg * self.params[f'W{i+1}']
            grads[f"b{i+1}"] = db 

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return loss, grads


def get_three_layer_network_params():
    weight_scale = 1e-2   # Experiment with this!
    learning_rate = 1e-4  # Experiment with this!
    ##########################################################################
    # TODO: Change weight_scale and learning_rate so your model achieves     #
    # 100% training accuracy within 20 epochs.                               #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    weight_scale = 1e-1
    learning_rate = 1e-1 
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return weight_scale, learning_rate


def get_five_layer_network_params():
    weight_scale = 1e-5   # Experiment with this!
    learning_rate = 2e-3  # Experiment with this!
    ##########################################################################
    # TODO: Change weight_scale and learning_rate so your model achieves     #
    # 100% training accuracy within 20 epochs.                               #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    weight_scale = 1e-1
    learning_rate = 1e-1 
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A tensor of the same shape as w and dw used to store
      a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    config.setdefault('velocity', torch.zeros_like(w))

    # Unpack parameters
    lr = config['learning_rate']
    m = config['momentum']
    v = config['velocity']

    next_w = None
    ##########################################################################
    # TODO: Implement the momentum update formula. Store the updated value   #
    # in the next_w variable. You should also update the velocity v.         #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    v = m * v - lr * dw 
    next_w = w + v 
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient dw_cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - dw_cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('dw_cache', torch.zeros_like(w))

    # Unpack parameters
    lr = config['learning_rate']
    rho = config['decay_rate']
    eps = config['epsilon']
    dw_cache = config['dw_cache']

    next_w = None
    ##########################################################################
    # TODO: Implement the RMSprop update formula. Store the next value of w  #
    # in the next_w variable. You should also update dw_cache.               #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    dw_cache = rho * dw_cache + (1 - rho) * dw * dw 
    next_w = w - lr * dw / (torch.sqrt(dw_cache) + eps)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    config['dw_cache'] = dw_cache

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both
    the gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    # Unpack parameters
    lr = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    eps = config['epsilon']
    m, v, t = config['m'], config['v'], config['t']

    next_w = None
    ##########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w   #
    # in next_w. You should also update m, v, and t.                         #
    # Note that, in order to match the reference output, please update t     #
    # before using it in any calculations.                                   #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    t += 1 
    m = beta1 * m + (1 - beta1) * dw 
    v = beta2 * v + (1 - beta2) * dw * dw 
    m_unbias = m / (1 - beta1 ** t)
    v_unbias = v / (1 - beta2 ** t)
    next_w = w - lr * m_unbias / (torch.sqrt(v_unbias) + eps)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    config['m'], config['v'], config['t'] = m, v, t

    return next_w, config


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for **inverted** dropout.
        Inputs:
        - x: A tensor containing input data, of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We **drop** each neuron output with
            the probability p.
          - mode: 'train' or 'test'. If the mode is train, perform dropout;
            if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Output; a tensor of the same shape as x
        - cache: Tuple (dropout_param, mask). In train mode, mask is the
          dropout mask that was used to multiply the input; in test mode,
          mask is None.
        NOTE: Keep in mind that p is the probability of **dropping**
              a neuron output; this might be contrary to some sources,
              where it is referred to as the probability of keeping a
              neuron output.
        """
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])

        p, mode = dropout_param['p'], dropout_param['mode']
        mask = None
        out = None

        if mode == 'train':
            ##################################################################
            # TODO: Implement training phase forward pass for **inverted**   #
            # dropout. Store the dropout mask in `mask`.                     #
            # Hint: torch.rand_like; Be aware of dtype.                      #
            ##################################################################
            # Replace "pass" with your code (do not modify this line)
            mask = (torch.rand_like(x) < (1-p)) / (1-p)
            out = x * mask  
            ##################################################################
            #                        END OF YOUR CODE                        #
            ##################################################################
        elif mode == 'test':
            ##################################################################
            # TODO: Implement the test phase forward pass for **inverted**   #
            # dropout.                                                       #
            ##################################################################
            # Replace "pass" with your code (do not modify this line)
            out = x 
            ##################################################################
            #                        END OF YOUR CODE                        #
            ##################################################################

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for **inverted** dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            ##################################################################
            # TODO: Implement training phase backward pass for **inverted**  #
            # dropout.                                                       #
            ##################################################################
            # Replace "pass" with your code (do not modify this line)
            dx = dout * mask
            ##################################################################
            #                        END OF YOUR CODE                        #
            ##################################################################
        elif mode == 'test':
            dx = dout
        return dx
