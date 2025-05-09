"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
You are NOT allowed to use torch.nn ops, unless otherwise specified.
"""
import torch

from a3_helpers import softmax_loss
from common import Solver
from fully_connected_networks import sgd_momentum, rmsprop, adam


def hello():
    """
    This is a sample function that we will try to import and run to ensure
    that our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modify the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ######################################################################
        # TODO: Implement the convolutional forward pass.                    #
        # Hint: You can use function torch.nn.functional.pad for padding.    #
        # You are NOT allowed to use anything in torch.nn in other places.   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        (N, C, H, W) = x.shape
        (F, C, HH, WW) = w.shape

        stride = conv_param["stride"]
        pad = conv_param["pad"]

        n_H = int( (H - HH + (2*pad)) / stride ) + 1
        n_W = int( (W - WW + (2*pad)) / stride) + 1 

        out = torch.zeros((N, F, n_H, n_W), dtype=x.dtype, device=x.device)

        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0)

        for i in range(N):
            x_pad_i = x_pad[i]

            for h in range(n_H):
                vert_start = stride * h 
                vert_end = vert_start + HH 

                for j in range(n_W):
                    horiz_start = stride * j 
                    horiz_end = horiz_start + WW 

                    for f in range(F):

                        slice = x_pad_i[:, vert_start:vert_end, horiz_start:horiz_end]

                        weights = w[f, :, :, :]
                        biases = b[f]

                        out[i, f, h, j] = torch.sum(torch.multiply(slice, weights)) + torch.squeeze(biases)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in Conv.forward

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ######################################################################
        # TODO: Implement the convolutional backward pass.                   #
        # Hint: You can use function torch.nn.functional.pad for padding.    #
        # You are NOT allowed to use anything in torch.nn in other places.   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        (x, w, b, conv_param) = cache 
        (N, C, H, W) = x.shape 
        (F, C, HH, WW) = w.shape 

        stride = conv_param["stride"]
        pad = conv_param["pad"]

        (N, F, n_H, n_W) = dout.shape 

        dx = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        dw = torch.zeros(w.shape, dtype=w.dtype, device=x.device)
        db = torch.zeros(b.shape, dtype=b.dtype, device=x.device)

        X_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
        dX_pad = torch.nn.functional.pad(dx, (pad, pad, pad, pad), mode='constant', value=0)

        for i in range(N):

            x_pad = X_pad[i]
            dx_pad = dX_pad[i]

            for h in range(n_H):
                for j in range(n_W):
                    for f in range(F):

                        vert_start = stride * h 
                        vert_end = vert_start + HH 
                        horiz_start = stride * j 
                        horiz_end = horiz_start + WW 

                        slice = x_pad[:, vert_start:vert_end, horiz_start:horiz_end]

                        dx_pad[:, vert_start:vert_end, horiz_start:horiz_end] += w[f, :,:,:] * dout[i, f, h, j]
                        dw[f, :,:,:] += slice * dout[i, f, h, j]
                        db[f] += dout[i, f, h, j]
            
            dx[i, :, :, :] = dx_pad[:, pad:-pad, pad:-pad]
        
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ######################################################################
        # TODO: Implement the max-pooling forward pass.                      #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        (N, C, H, W) = x.shape 

        ph = pool_param["pool_height"]
        pw = pool_param["pool_width"]
        stride = pool_param["stride"]

        n_H = int( 1 + (H - ph) / stride)
        n_W = int( 1 + (W - pw) / stride )

        out = torch.zeros((N, C, n_H, n_W), dtype=x.dtype, device=x.device)

        for i in range(N):
            x_i = x[i]
            for h in range(n_H):
                vert_start = stride * h 
                vert_end = vert_start + ph 

                for j in range(n_W):
                    horiz_start = stride * j 
                    horiz_end = horiz_start + pw 

                    for c in range(C):
                        x_slice = x_i[c, vert_start:vert_end, horiz_start:horiz_end]

                        out[i, c, h, j] = torch.max(x_slice)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        ######################################################################
        # TODO: Implement the max-pooling backward pass.                     #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        (x, pool_param) = cache 

        stride = pool_param["stride"]
        ph = pool_param["pool_height"]
        pw = pool_param["pool_width"]

        (N, C, H, W) = x.shape 
        (N, C, n_H, n_W) = dout.shape 

        dx = torch.zeros(x.shape, dtype=dout.dtype, device=x.device)

        for i in range(N):

            x_i = x[i, :, :, :]

            for h in range(n_H):
                for j in range(n_W):
                    for c in range(C):

                        vert_start = h * stride 
                        vert_end = vert_start + ph 
                        horiz_start = j * stride 
                        horiz_end = horiz_start + pw 

                        x_slice = x_i[c, vert_start:vert_end, horiz_start:horiz_end]
                        mask = (x_slice == torch.max(x_slice))

                        dx[i, c, vert_start:vert_end, horiz_start:horiz_end] += mask * dout[i, c, h, j]
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights and biases for three-layer convolutional  #
        # network. Weights should be initialized from the Gaussian           #
        # distribution with the mean of 0.0 and the standard deviation of    #
        # weight_scale; biases should be initialized to zero. All weights    #
        # and biases should be stored in the dictionary self.params.         #
        # Store weights and biases for the convolutional layer using the     #
        # keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the weights and     #
        # biases of the hidden linear layer, and keys 'W3' and 'b3' for the  #
        # weights and biases of the output linear layer.                     #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        F, (C, H, W) = num_filters, input_dims 

        self.device = 'cuda'

        self.params['W1'] = torch.randn((F, C, filter_size, filter_size), dtype=dtype, device=self.device) * weight_scale
        self.params['b1'] = torch.zeros(num_filters, dtype=dtype, device=self.device)
        self.params['W2'] = torch.randn((F * H * W // 4, hidden_dim), dtype=dtype, device=self.device) * weight_scale
        self.params['b2'] = torch.zeros(hidden_dim, dtype=dtype, device=self.device)
        self.params['W3'] = torch.randn((hidden_dim, num_classes), dtype=dtype, device=self.device) * weight_scale
        self.params['b3'] = torch.zeros(num_classes, dtype=dtype, device=self.device)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Pass conv_param to the forward pass for the convolutional layer.
        # Padding and stride chosen to preserve the input spatial size.
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # Pass pool_param to the forward pass for the max-pooling layer.
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        # Use sandwich layers if Linear or Conv layers followed by ReLU      #
        # and/or Pool layers for efficient implementation.                   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        out, cache_1 = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        out, cache_2 = Linear_ReLU.forward(out, W2, b2)
        scores, cache_3 = Linear.forward(out, W3, b3)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ######################################################################
        # TODO: Implement the backward pass for three-layer convolutional    #
        # net, storing the loss and gradients in the loss and grads.         #
        # Compute the data loss using softmax, and make sure that grads[k]   #
        # holds the gradients for self.params[k]. Don't forget to add        #
        # L2 regularization!                                                 #
        # NOTE: To ensure your implementation matches ours and you pass the  #
        # automated tests, make sure that your L2 regularization includes    #
        # a factor of 0.5 to simplify the expression for the gradient.       #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * torch.tensor([torch.sum(W**2) for k, W in self.params.items() if 'W' in k]).sum()

        dout, dw, db = Linear.backward(dout, cache_3)
        grads['W3'] = dw + self.reg * W3
        grads['b3'] = db 

        dout, dw, db = Linear_ReLU.backward(dout, cache_2)
        grads['W2'] = dw + self.reg * W2
        grads['b2'] = db 

        dout, dw, db = Conv_ReLU_Pool.backward(dout, cache_1)
        grads['W1'] = dw + self.reg * W1 
        grads['b1'] = db 
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string 'kaiming' to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this dtype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == torch.device('cuda'):
            device = torch.device('cuda:0')

        ######################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights,  #
        # biases, and batchnorm scale and shift parameters should be stored  #
        # in the dictionary self.params, where the keys should be in the     #
        # form of 'W#', 'b#', 'gamma#', and 'beta#' with 1-based indexing.   #
        # Weights for Conv and Linear layers should be initialized from the  #
        # Gaussian distribution with the mean of 0.0 and the standard        #
        # deviation of weight_scale; however, if weight_scale == 'kaiming',  #
        # then you should call kaiming_initializer instead. Biases should be #
        # initialized to zeros. Batchnorm scale (gamma) and shift (beta)     #
        # parameters should be initialized to ones and zeros, respectively.  #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        C, H, W = input_dims 

        filter_dims = [input_dims[0], *num_filters] # [3, 8, 8, 8, 8, 8]

        for i in range(self.num_layers-1):
            if weight_scale == 'kaiming':
                self.params[f'W{i+1}'] = kaiming_initializer(Din=filter_dims[i], Dout=filter_dims[i+1], K=3, relu=True, dtype=self.dtype, device=device)
            else:
                self.params[f'W{i+1}'] = torch.randn((filter_dims[i+1], filter_dims[i], 3, 3), dtype=self.dtype, device=device) * weight_scale
            self.params[f'b{i+1}'] = torch.zeros(filter_dims[i+1], dtype=self.dtype, device=device)
        
        if weight_scale == 'kaiming':
            self.params[f'W{self.num_layers}'] = kaiming_initializer(Din=filter_dims[-1] * H * W // (4**len(max_pools)), Dout=num_classes, K=None, relu=False, dtype=self.dtype, device=device)
        else:
            self.params[f'W{self.num_layers}'] = torch.randn((filter_dims[-1] * H * W // (4**len(max_pools)), num_classes), dtype=self.dtype, device=device)
        self.params[f'b{self.num_layers}'] = torch.zeros(num_classes, dtype=self.dtype, device=device)

        if self.batchnorm:
            for i in range(self.num_layers-1):
                self.params[f"gamma{i+1}"] = torch.ones(filter_dims[i+1], dtype=self.dtype, device=device)
                self.params[f'beta{i+1}'] = torch.zeros(filter_dims[i+1], dtype=self.dtype, device=device)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for DeepConvNet, computing the    #
        # class scores for X and storing them in the scores variable.        #
        # Use sandwich layers if Linear or Conv layers followed by ReLU      #
        # and/or Pool layers for efficient implementation.                   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        # conv-relu-pool or conv-relu 
        cache = {}
        
        for i in range(self.num_layers-1):
            if i in self.max_pools:
                if self.batchnorm:
                    X, cache[i] = Conv_BatchNorm_ReLU_Pool.forward(
                        X, self.params[f'W{i+1}'], self.params[f'b{i+1}'], 
                        self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], 
                        conv_param, bn_param, pool_param)
                else:
                    X, cache[i] = Conv_ReLU_Pool.forward(X, self.params[f"W{i+1}"], self.params[f'b{i+1}'], conv_param, pool_param)
            else:
                if self.batchnorm:
                    X, cache[i] = Conv_BatchNorm_ReLU.forward(
                        X, self.params[f'W{i+1}'], self.params[f'b{i+1}'], 
                        self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'],
                        conv_param, bn_param)
                else:
                    X, cache[i] = Conv_ReLU.forward(X, self.params[f"W{i+1}"], self.params[f'b{i+1}'], conv_param)
        
        scores, cache[self.num_layers-1] = Linear.forward(X, self.params[f"W{self.num_layers}"], self.params[f'b{self.num_layers}'])

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ######################################################################
        # TODO: Implement the backward pass for the DeepConvNet, storing the #
        # loss and gradients in the loss and grads variables.                #
        # Compute the data loss using softmax, and make sure that grads[k]   #
        # holds the gradients for self.params[k]. Don't forget to add        #
        # L2 regularization!                                                 #
        # NOTE: To ensure your implementation matches ours and you pass the  #
        # automated tests, make sure that your L2 regularization includes    #
        # a factor of 0.5 to simplify the expression for the gradient.       #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * torch.tensor([torch.sum(W**2) for k, W in self.params.items() if 'W' in k]).sum()

        dout, dw, db = Linear.backward(dout, cache[self.num_layers-1])

        grads[f'W{self.num_layers}'] = dw + self.reg * self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = db 

        for i in reversed(range(self.num_layers-1)):
            if i in self.max_pools:
                if self.batchnorm:
                    dout, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(dout, cache[i])
                else:
                    dout, dw, db = Conv_ReLU_Pool.backward(dout, cache[i])
            else:
                if self.batchnorm:
                    dout, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(dout, cache[i])
                else:
                    dout, dw, db = Conv_ReLU.backward(dout, cache[i])
            
            grads[f'W{i+1}'] = dw + self.reg * self.params[f'W{i+1}']
            grads[f'b{i+1}'] = db 

            if self.batchnorm:
                grads[f'gamma{i+1}'] = dgamma
                grads[f'beta{i+1}'] = dbeta 

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ##########################################################################
    # TODO: Change weight_scale and learning_rate so your model achieves     #
    # 100% training accuracy within 30 epochs.                               #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    weight_scale = 1e-1
    learning_rate = 1e-3 
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return weight_scale, learning_rate


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initialization); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ######################################################################
        # TODO: Implement the Kaiming initialization for linear layer.       #
        # The weight_scale is sqrt(gain / fan_in), where gain is 2 if ReLU   #
        # is followed by the layer, or 1 if not, and fan_in = Din.           #
        # The output should be a tensor in the designated size, dtype,       #
        # and device.                                                        #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        weight = torch.randn(Din, Dout, dtype=dtype, device=device) * torch.tensor(gain / Din).sqrt()
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    else:
        ######################################################################
        # TODO: Implement Kaiming initialization for convolutional layer.    #
        # The weight_scale is sqrt(gain / fan_in), where gain is 2 if ReLU   #
        # is followed by the layer, or 1 if not, and fan_in = Din * K * K.   #
        # The output should be a tensor in the designated size, dtype,       #
        # and device.                                                        #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        weight = torch.randn((Dout, Din, K, K), dtype=dtype, device=device) * torch.tensor(gain / (Din * K * K)).sqrt()
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    return weight


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    ##########################################################################
    # TODO: Train the best DeepConvNet on CIFAR-10 within 30 seconds.        #
    # Hint: You can use any optimizer you implemented in                     #
    # fully_connected_networks.py, which we imported for you.                #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    input_dims = data_dict['X_train'].shape[1:]
    model = DeepConvNet(input_dims=input_dims, num_classes=10,
                        num_filters=[64] * 3, max_pools=[0, 1, 2],
                        weight_scale='kaiming', reg=1e-5, device=device)
    
    solver = Solver(model, data_dict, num_epochs=9, batch_size=128,
                    optim_config={
                        'learning_rate':1e-3
                    }, print_every=40, device='cuda', update_rule=adam)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return solver

class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each time step, we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift parameter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = \
            bn_param.get('running_mean',
                         torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = \
            bn_param.get('running_var',
                         torch.ones(D, dtype=x.dtype, device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batchnorm.  #
            # Use minibatch statistics to compute the mean and variance.     #
            # Use the mean and variance to normalize the incoming data, and  #
            # then scale and shift the normalized data using gamma and beta. #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" with your code (do not modify this line)
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            std = torch.sqrt(var + eps)
            x_hat = (x - mu) / std 
            out = gamma * x_hat + beta 
            cache = (x, mu, var, std, gamma, x_hat)

            running_mean = momentum * running_mean + (1 - momentum) * mu 
            running_var = momentum * running_var + (1 - momentum) * var 
            ##################################################################
            #                        END OF YOUR CODE                        #
            ##################################################################
        elif mode == 'test':
            ##################################################################
            # TODO: Implement the test-time forward pass for batchnorm.      #
            # Use the running mean and variance to normalize the incoming    #
            # data, and then scale and shift the normalized data using gamma #
            # and beta. Store the result in the out variable.                #
            ##################################################################
            # Replace "pass" with your code (do not modify this line)
            x_hat = (x - running_mean) / torch.sqrt(running_var + eps)
            out = gamma * x_hat + beta 
            ##################################################################
            #                        END OF YOUR CODE                        #
            ##################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        ######################################################################
        # TODO: Implement the backward pass for batch normalization.         #
        # Store the results in the dx, dgamma, and dbeta variables.          #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)  #
        # might prove to be helpful.                                         #
        # Don't forget to implement train and test mode separately.          #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        x, mu, var, std, gamma, x_hat = cache 

        dbeta = dout.sum(axis=0)
        dgamma = (dout * x_hat).sum(axis=0)

        dx_hat = dout * gamma
        dstd = -torch.sum(dx_hat * (x-mu), axis=0) / (std**2)
        dvar = 0.5 * dstd / std
        dx1 = dx_hat / std + 2 * (x-mu) * dvar / len(dout)
        dmu = -torch.sum(dx1, axis=0)
        dx2 = dmu / len(dout)
        dx = dx1 + dx2 
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalization backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ######################################################################
        # TODO: Implement the backward pass for batch normalization.         #
        # Store the results in the dx, dgamma, and dbeta variables.          #
        #                                                                    #
        # Note: after computing the gradient with respect to the centered    #
        # inputs, gradients with respect to the inputs (dx) can be written   #
        # in a single statement; our implementation fits on a single         #
        # 80-character line. But, it is okay to write it in multiple lines.  #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        x, mu, var, std, gamma, x_hat = cache
        S = lambda x: x.sum(axis=0)

        dbeta = dout.sum(axis=0)
        dgamma = (dout * x_hat).sum(axis=0)

        dx = dout * gamma / (len(dout) * std)
        dx = len(dout) * dx - S(dx*x_hat)*x_hat - S(dx)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ######################################################################
        # TODO: Implement the forward pass for spatial batch normalization.  #
        # You should implement this by calling the 1D batch normalization    #
        # you implemented above with permuting and/or reshaping input/output #
        # tensors. Your implementation should be very short;                 #
        # less than five lines are expected.                                 #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        N, C, H, W = x.shape
        x = torch.moveaxis(x, 1, -1).reshape(-1, C)
        out, cache = BatchNorm.forward(x, gamma, beta, bn_param)
        out = torch.moveaxis(out.reshape(N, H, W, C), -1, 1)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        ######################################################################
        # TODO: Implement the backward pass for spatial batch normalization. #
        # You should implement this by calling the 1D batch normalization    #
        # you implemented above with permuting and/or reshaping input/output #
        # tensors. Your implementation should be very short;                 #
        # less than five lines are expected.                                 #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        N, C, H, W = dout.shape 
        dout = torch.moveaxis(dout, 1, -1).reshape(-1, C)
        dx, dgamma, dbeta = BatchNorm.backward(dout, cache)
        dx = torch.moveaxis(dx.reshape(N, H, W, C), -1, 1)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return dx, dgamma, dbeta


##################################################################
#            Fast Implementations and Sandwich Layers            #
##################################################################


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        layer = torch.nn.Linear(*w.shape)
        layer.weight = torch.nn.Parameter(w.T)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx.flatten(start_dim=1))
        cache = (x, w, b, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, w, b, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach().T
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dw = torch.zeros_like(layer.weight).T
            db = torch.zeros_like(layer.bias)
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        layer = torch.nn.ReLU()
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
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


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dw = torch.zeros_like(layer.weight)
            db = torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool convenience layer.
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class FastBatchNorm(object):
    func = torch.nn.BatchNorm1d

    @classmethod
    def forward(cls, x, gamma, beta, bn_param):
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)
        D = x.shape[1]
        running_mean = \
            bn_param.get('running_mean',
                         torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = \
            bn_param.get('running_var',
                         torch.ones(D, dtype=x.dtype, device=x.device))

        layer = cls.func(D, eps=eps, momentum=momentum,
                         device=x.device, dtype=x.dtype)
        layer.weight = torch.nn.Parameter(gamma)
        layer.bias = torch.nn.Parameter(beta)
        layer.running_mean = running_mean
        layer.running_var = running_var
        if mode == 'train':
            layer.train()
        elif mode == 'test':
            layer.eval()
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (mode, x, tx, out, layer)
        # Store the updated running means back into bn_param
        bn_param['running_mean'] = layer.running_mean.detach()
        bn_param['running_var'] = layer.running_var.detach()
        return out, cache

    @classmethod
    def backward(cls, dout, cache):
        mode, x, tx, out, layer = cache
        try:
            if mode == 'train':
                layer.train()
            elif mode == 'test':
                layer.eval()
            else:
                raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
            out.backward(dout)
            dx = tx.grad.detach()
            dgamma = layer.weight.grad.detach()
            dbeta = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dgamma = torch.zeros_like(layer.weight)
            dbeta = torch.zeros_like(layer.bias)
        return dx, dgamma, dbeta


class FastSpatialBatchNorm(FastBatchNorm):
    func = torch.nn.BatchNorm2d


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D1, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = FastBatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = FastBatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = FastSpatialBatchNorm.forward(a, gamma,
                                                    beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = FastSpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = FastSpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = FastSpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
