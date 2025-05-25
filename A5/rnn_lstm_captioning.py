"""
Implements rnn and lstm for image captioning in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import math
from typing import Optional, Tuple

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import feature_extraction


def hello():
    """
    This is a sample function that we will try to import and run to ensure
    that our environment is correctly set up on Google Colab.
    """
    print('Hello from rnn_lstm_captioning.py!')


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses
    a tanh activation function.

    The input data has dimension D, the hidden state has dimension H, and we
    use a minibatch size of N.

    Args:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases, of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##########################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store next  #
    # hidden state and any values you need for the backward pass in next_h   #
    # and cache variables respectively.                                      #
    # Hint: torch.tanh                                                       #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    val = x @ Wx + prev_h @ Wh + b
    next_h = torch.tanh(val)

    cache = (x, prev_h, Wx, Wh, val)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Args:
    - dnext_h: Gradient of loss with respect to next hidden state,
      of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement a single backward step of a vanilla RNN.               #
    # Hint: d tanh(x) / dx = 1 - [tanh(x)]^2                                 #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    x, prev_h, Wx, Wh, val = cache # (N, D), (N, H), (D, H), (H, H), (N, H)

    dval = dnext_h * (1 - torch.tanh(val) ** 2) # (N, H)

    dx = dval @ Wx.T # (N, D)
    dprev_h = dval @ Wh.T # (N, H)
    dWx = x.T @ dval # (D, H)
    dWh = prev_h.T @ dval # (H, H)
    db = dval.sum(dim=0) # (H, )

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After
    running the RNN forward, we return the hidden states for all timesteps.

    Args:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases, of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##########################################################################
    # TODO: Implement the forward pass for a vanilla RNN running on a        #
    # sequence of input data. You should use the rnn_step_forward function   #
    # that you defined above.                                                #
    # Hint: You may want to use a for-loop.                                  #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    N, T, D = x.shape
    H = h0.shape[1]

    h = torch.zeros((N, T, H), device=x.device, dtype=x.dtype)
    cache = []

    prev_h = h0
    for t in range(T):
        xt = x[:, t, :] # (N, D)
        next_h, step_cache = rnn_step_forward(xt, prev_h, Wx, Wh, b)
        h[:, t, :] = next_h
        cache.append(step_cache)
        prev_h = next_h
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Backward pass for a vanilla RNN over an entire sequence of data.

    Args:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H).

    NOTE: 'dh' contains the upstream gradients produced by the
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a vanilla RNN running on a       #
    # sequence of input data. You should use the rnn_step_backward function  #
    # that you defined above.                                                #
    # Hint: You may want to use a for-loop.                                  #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    N, T, H = dh.shape
    x0 = cache[0][0]
    D = x0.shape[1]

    dx = torch.zeros((N, T, D), device=dh.device, dtype=dh.dtype)
    dWx = torch.zeros((D, H), device=dh.device, dtype=dh.dtype)
    dWh = torch.zeros((H, H), device=dh.device, dtype=dh.dtype)
    db = torch.zeros((H,), device=dh.device, dtype=dh.dtype)
    dprev_h = torch.zeros((N, H), device=dh.device, dtype=dh.dtype)

    for t in reversed(range(T)):
        dnext_h = dh[:, t, :] + dprev_h 
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h, cache[t])
        dx[:, t, :] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t 
    
    dh0 = dprev_h
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return dx, dh0, dWx, dWh, db


class RNN(nn.Module):
    """
    Single-layer vanilla RNN module.

    You don't have to implement anything here but it is highly recommended to
    read through the code as you will implement subsequent modules.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize an RNN. Model parameters to initialize:
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases, of shape (H,)

        Args:
        _ input_dim: Input size, denoted as D before
        _ hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        """
        Args:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - h0: Initial hidden state, of shape (N, H)

        Returns:
        - hn: The hidden state output
        """
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        """
        Args:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)

        Returns:
        - next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h


class ImageEncoder(nn.Module):
    """
    Convolutional network that accepts images as input and outputs their
    spatial grid features. This module servesx as the image encoder in image
    captioning model. We will use a tiny RegNet-X 400MF model that is
    initialized with ImageNet-pretrained weights from Torchvision library.

    NOTE: We could use any convolutional network architecture, but we opt for
    a tiny RegNet model so it can train decently with a single Tesla T4 GPU.
    """
    def __init__(self, pretrained: bool = True, verbose: bool = True):
        """
        Args:
        - pretrained: Whether to initialize this model with pretrained
          weights from Torchvision library.
        - verbose: Whether to log expected output shapes during instantiation
        """
        super().__init__()
        if pretrained:
            weights = torchvision.models.RegNet_X_400MF_Weights.IMAGENET1K_V2
        else:
            weights = None
        self.cnn = torchvision.models.regnet_x_400mf(weights=weights)

        # Torchvision models return global average pooled features by default.
        # Our attention-based models may require spatial grid features. So we
        # wrap the ConvNet with torchvision's feature extractor. We will get
        # the spatial features right before the final classification layer.
        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={'trunk_output.block4': 'c5'}
        )
        # We call these features "c5", a name that may sound familiar from the
        # object detection assignment. :-)

        # Pass a dummy batch of input images to infer output shape.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))['c5']
        self._out_channels = dummy_out.shape[1]

        if verbose:
            print('For input images in NCHW format, shape (2, 3, 224, 224)')
            print(f'Shape of output c5 features: {dummy_out.shape}')

        # Input image batches are expected to be float tensors in range [0, 1].
        # However, the backbone here expects these tensors to be normalized by
        # ImageNet color mean/std (as it was trained that way).
        # We define a function to transform the input images before extraction:
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        """
        Number of output channels in extracted image features. You may access
        this value freely to define more modules to go with this encoder.
        """
        return self._out_channels

    def forward(self, images: torch.Tensor):
        # Input images may be uint8 tensors in [0-255], change them to float
        # tensors in [0-1]. Get float type from backbone (could be float32/64).
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0

        # Normalize images by ImageNet color mean/std.
        images = self.normalize(images)

        # Extract c5 features from encoder (backbone) and return.
        # shape: (B, out_channels, H / 32, W / 32)
        features = self.backbone(images)['c5']
        return features


class WordEmbedding(nn.Module):
    """
    Simplified version of torch.nn.Embedding.

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning
    each word to a vector of dimension D.

    Args:
    - x: Integer array of shape (N, T) giving indices of words. Each element
      idx of x must be in the range 0 <= idx < V.

    Returns:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    """
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()

        # Register parameters
        self.W_embed = nn.Parameter(
            torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size))
        )

    def forward(self, x):
        out = None
        ######################################################################
        # TODO: Implement the forward pass for word embeddings.              #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        out = self.W_embed[x]
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out


def temporal_softmax_loss(x, y, ignore_index=None):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives
    scores for all vocabulary elements at all timesteps, and y gives the
    indices of the ground-truth element at each timestep. We use a
    cross-entropy loss at each timestep, *summing* the loss over all timesteps
    and *averaging* across the minibatch.

    As an additional complication, we may want to ignore the model output at
    some timesteps, since sequences of different length may have been combined
    into a minibatch and padded with NULL tokens. The optional ignore_index
    argument tells us which elements in the caption should not contribute to
    the loss.

    Args:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the
      range 0 <= y[i, t] < V

    Returns:
    - loss: Scalar giving loss
    """
    loss = None
    ##########################################################################
    # TODO: Implement the temporal softmax loss function. Note that we       #
    # compute the cross-entropy loss at each timestep, summing the loss over #
    # all timesteps and averaging across the minibatch.                      #
    # Hint: F.cross_entropy(..., ignore_index, reduction)                    #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    N, T, V = x.shape
    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)

    loss_flat = F.cross_entropy(x_flat, y_flat, ignore_index=ignore_index, reduction="none") # (N*T,)

    if ignore_index is not None:
        mask = (y_flat != ignore_index)
        loss_flat = loss_flat * mask 
    
    # Reshape back to (N, T) and sum over time, then average over batch
    loss = loss_flat.reshape(N, T).sum(dim=1).mean()
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss


class CaptioningRNN(nn.Module):
    """
    A CaptioningRNN produces captions from images using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.

    You will implement the `__init__` method for model initialization and
    the `forward` method first, then come back for the `sample` method later.
    """
    def __init__(
        self,
        word_to_idx,
        input_dim: int = 512,
        wordvec_dim: int = 128,
        hidden_dim: int = 128,
        cell_type: str = 'rnn',
        image_encoder_pretrained: bool = True,
        ignore_index: Optional[int] = None,
    ):
        """
        Construct a new CaptioningRNN instance.

        Args:
        - word_to_idx: A dictionary giving the vocabulary. It contains V
          entries, and maps each string to a unique integer in the range [0, V)
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        """
        super().__init__()
        if cell_type not in {'rnn', 'lstm', 'attn'}:
            raise ValueError("Invalid cell_type '%s'" % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        self.ignore_index = ignore_index

        self.image_encoder = None
        self.feat_proj = None
        self.word_embed = None
        self.rnn = None
        self.output_proj = None
        ######################################################################
        # TODO: Initialize the image captioning module by defining:          #
        # self.image_encoder using ImageEncoder                              #
        # self.feat_proj as a 1x1 conv from CNN pooled feature to `h0`       #
        # by global average pooling - Conv2d - flatten for RNN and LSTM, or  #
        # by Conv2d for AttentionLSTM                                        #
        # self.word_embed using WordEmbedding                                #
        # self.rnn using RNN, LSTM, or AttentionLSTM depending on `cell_type`#
        # self.output_proj using nn.Linear (from RNN hidden state to vocab   #
        # probability)                                                       #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        self.image_encoder = ImageEncoder(pretrained=image_encoder_pretrained)

        if cell_type == "attn":
            self.feat_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        else:
            self.feat_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), # (N, D, 1, 1)
                nn.Conv2d(input_dim, hidden_dim, kernel_size=1), # (N, H, 1, 1)
                nn.Flatten(start_dim=1) # (N, H)
            )
        
        self.word_embed = WordEmbedding(vocab_size, wordvec_dim)

        if cell_type == "rnn":
            self.rnn = RNN(input_dim=wordvec_dim, hidden_dim=hidden_dim)
        elif cell_type == "lstm":
            self.rnn = LSTM(input_dim=wordvec_dim, hidden_dim=hidden_dim)
        elif cell_type == "attn":
            self.rnn = AttentionLSTM(input_dim=wordvec_dim, hidden_dim=hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(self, images, captions):
        """
        Compute training-time loss for the RNN. We input images and the GT
        captions for those images, and use an RNN (or LSTM) to compute loss.
        The backward part will be done by torch.autograd.

        Args:
        - images: Input images, of shape (N, 3, 112, 112)
        - captions: Ground-truth captions; an integer array of shape
          (N, T + 1) where each element is in the range 0 <= y[i, t] < V

        Returns:
        - loss: A scalar loss
        """
        """
        Cut captions into two pieces: captions_in has everything but the last
        word and will be input to the RNN; captions_out has everything but the
        first word and this is what we will expect the RNN to generate. These
        are offset by one relative to each other because the RNN should produce
        word (t+1) after receiving word t. The first element of captions_in
        will be the START token, and the first element of captions_out will
        be the first word.
        """
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        loss = 0.0
        ######################################################################
        # TODO: Implement the forward pass for the CaptioningRNN.            #
        # In the forward pass you will need to do the following:             #
        # 1) Use self.word_embed to transform words in `captions_in` from    #
        # indices to vectors, giving an array of shape (N, T, W).            #
        # 2) Extract image features using self.image_encoder                 #
        # 3) Use self.feat_proj to project the image feature to              #
        # the initial hidden state `h0` (for RNN/LSTM, of shape (N, H))      #
        # or the projected CNN activation input `A` (for Attention LSTM,     #
        # of shape (N, H, D_a, D_a).)                                        #
        # 4) Use self.rnn to process the sequence of input word vectors and  #
        # produce hidden state vectors for all timesteps, producing an array #
        # of shape (N, T, H).                                                #
        # 5) Apply self.output_proj to compute scores over the vocabulary at #
        # every timestep from the hidden states, giving an array             #
        # of shape (N, T, V).                                                #
        # 6) Use (temporal) softmax to compute loss using captions_out,      #
        # ignoring the points where the output word is <NULL>.               #
        # Do not worry about regularizing the weights or their gradients!    #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        # 1) Embed input words (N, T, W)
        word_vectors = self.word_embed(captions_in) 

        # 2) Extract image features 
        features = self.image_encoder(images)

        # 3) Project image features to initial hidden state h0 
        # 4) Process seq of input word vectors and produce hidden state vectors (N, T, H)
        if self.cell_type == "attn":
            A = self.feat_proj(features) # (N, H, H', W')
            h = self.rnn(word_vectors, A) # (N, T, H)
        else:
            h0 = self.feat_proj(features) # (N, H)
            h = self.rnn(word_vectors, h0) # (N, T, H)
        
        # 5) Compute scores (N, T, V)
        scores = self.output_proj(h) # (N, T, V)

        # 6) Compute loss 
        loss = temporal_softmax_loss(scores, captions_out, ignore_index=self.ignore_index)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return loss

    def sample(self, images, max_length=16):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous
        hidden state to the RNN to get the next hidden state, use the hidden
        state to get scores for all vocab words, and choose the word with the
        highest score as the next word. The initial hidden state is computed by
        applying an affine transform to the image features, and the initial
        word is the <START> token.

        For LSTMs you will also have to keep track of the cell state; in that
        case the initial cell state should be zero.

        Args:
        - images: Input images, of shape (N, 3, 112, 112)
        - max_length: Maximum length T of generated captions

        Returns a tuple of:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V).
          The first element (captions[:, 0]) should be the <START> token.
        - attn_weights_all: Array of shape (N, max_length, D_a, D_a) giving
          attention weights returned only when self.cell_type == 'attn'
          The first attn weights (attn_weights_all[:, 0]) should be all zero.
        """
        N = images.shape[0]
        captions = torch.full((N, max_length), self._null,
                              dtype=torch.long, device=images.device)

        if self.cell_type == 'attn':
            D_a = 4
            attn_weights_all = \
                torch.zeros(N, max_length, D_a, D_a,
                            dtype=torch.float, device=images.device)
        ######################################################################
        # TODO: Implement test-time sampling for the model. You need to      #
        # initialize the hidden state `h0` by applying self.feat_proj to the #
        # image features. For LSTM, as we provided in LSTM forward function, #
        # you need to set the initial cell state `c0` to zero.               #
        # For AttentionLSTM, `c0 = h0 = A.mean(dim=(2, 3))`. The first word  #
        # that you feed to the RNN should be the <START> token; its value is #
        # stored in the variable self._start. After initial setting, at each #
        # time step, you need to do to:                                      #
        # 1) Embed the previous word using the learned word embeddings.      #
        # 2) Make an RNN step using the previous hidden state and the        #
        # embedded current word to get the next hidden state.                #
        # For visualization, store attention weights in `attn_weights_all`   #
        # if the model is AttentionLSTM.                                     #
        # 3) Apply the output projection to the next hidden state            #
        # to get scores for all words in the vocabulary.                     #
        # 4) Select the word with the highest score as the next word,        #
        # writing it (the word index) to the appropriate slot in the         #
        # captions variable.                                                 #
        # For simplicity, you do not need to stop generating after an <END>  #
        # token is sampled, but you can do so if you want.                   #
        # Hint: We are working over minibatches in this function.            #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        features = self.image_encoder(images)

        if self.cell_type == "attn":
            A = self.feat_proj(features) 
            h = A.mean(dim=(2, 3))
            c = torch.zeros_like(h)
        else:
            h = self.feat_proj(features)
            if self.cell_type == "lstm":
                c = torch.zeros_like(h)
        
        captions[:, 0] = self._start 
        prev_word = torch.full((N,), self._start, dtype=torch.long, device=images.device)

        for t in range(1, max_length):
            word_embed = self.word_embed(prev_word)

            if self.cell_type == "rnn":
                Wx, Wh, b = self.rnn.Wx, self.rnn.Wh, self.rnn.b
                h = torch.tanh(word_embed @ Wx + h @ Wh + b)
            elif self.cell_type == "lstm":
                Wx, Wh, b = self.rnn.Wx, self.rnn.Wh, self.rnn.b 
                a = word_embed @ Wx + h @ Wh + b 
                H = h.shape[1]
                i = torch.sigmoid(a[:, :H])
                f = torch.sigmoid(a[:, H:2*H])
                o = torch.sigmoid(a[:, 2*H:3*H])
                g = torch.tanh(a[:, 3*H:])
                c = f * c + i * g
                h = o * torch.tanh(c)
            elif self.cell_type == "attn":
                A_flat = A.view(N, h.shape[1], -1)
                scores = (h.unsqueeze(1) @ A_flat).squeeze(1) / math.sqrt(h.shape[1])
                attn_weights = torch.softmax(scores, dim=1).view(N, D_a, D_a)
                attn = (A_flat @ attn_weights.view(N, -1, 1)).squeeze(2)

                Wx, Wh, Wattn, b = self.rnn.Wx, self.rnn.Wh, self.rnn.Wattn, self.rnn.b 
                a = word_embed @ Wx + h @ Wh + attn @ Wattn + b
                H = h.shape[1]
                i = torch.sigmoid(a[:, :H])
                f = torch.sigmoid(a[:, H : 2*H])
                o = torch.sigmoid(a[:, 2*H : 3*H])
                g = torch.tanh(a[:, 3*H:])
                c = f * c + i * g
                h = o * torch.tanh(c)

                attn_weights_all[:, t] = attn_weights 
            
            scores = self.output_proj(h)
            _, next_word = scores.max(dim=1)
            captions[:, t] = next_word
            prev_word = next_word
        
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        if self.cell_type == 'attn':
            return captions, attn_weights_all.cpu()
        else:
            return captions


class LSTM(nn.Module):
    """
    Single-layer, uni-directional LSTM module.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Args:
        - input_dim: Input size, denoted as D before
        - hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep of an LSTM.
        The input data has dimension D, the hidden state has dimension H, and
        we use a minibatch size of N.

        Args:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)
        - prev_c: The previous cell state, of shape (N, H)
        - Wx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        """
        next_h, next_c = None, None
        ######################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM. #
        # Hint: torch.sigmoid, torch.tanh                                    #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        h = prev_h.shape[1]

        a = x @ self.Wx + prev_h @ self.Wh + self.b # (N, 4H)
        ai, af, ao, ag = torch.chunk(a, 4, dim=1) # split into 4 parts

        i = torch.sigmoid(ai) # input gate
        f = torch.sigmoid(af) # forget gate
        o = torch.sigmoid(ao) # output gate
        g = torch.tanh(ag) # block input 

        next_c = f * prev_c + i * g # cell update
        next_h = o * torch.tanh(next_c) # hidden state 
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return next_h, next_c

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM
        uses a hidden size of H, and we work over a minibatch containing N
        sequences. After running the LSTM forward, we return the hidden states
        for all timesteps.

        Note that the initial cell state is set to zero, and the final cell
        state is not returned; it is an internal variable to the LSTM and is
        not accessed from outside.

        Args:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - h0: Initial hidden state, of shape (N, H)

        Returns:
        - hn: The hidden state output.
        """
        c0 = torch.zeros_like(h0)  # initial cell state
        hn = None
        ######################################################################
        # TODO: Implement the forward pass for an LSTM running on a sequence #
        # of input data.                                                     #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        N, T, D = x.shape
        H = h0.shape[1]

        hn = torch.zeros(N, T, H, device=x.device, dtype=x.dtype)
        h, c = h0, c0 

        for t in range(T):
            xt = x[:, t, :] # (N, D)
            h, c = self.step_forward(xt, h, c)
            hn[:, t, :] = h 
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return hn


def dot_product_attention(prev_h, A):
    """
    A simple scaled dot-product attention layer.

    Args:
    - prev_h: The LSTM hidden state from previous time step, of shape (N, H)
    - A: **Projected** CNN feature activation, of shape (N, H, D_a, D_a),
       where H is the LSTM hidden state size

    Returns a tuple of:
    - attn: Attention embedding output, of shape (N, H)
    - attn_weights: Attention weights, of shape (N, D_a, D_a)

    """
    N, H, D_a, _ = A.shape

    attn, attn_weights = None, None
    ##########################################################################
    # TODO: Implement the scaled dot-product attention we described earlier. #
    # HINT: torch.bmm, torch.softmax                                         #
    # Make sure you reshape `attn_weights` back to (N, D_a, D_a).            #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    A_flat = A.reshape(N, H, D_a * D_a) # (N, H, D_a * D_a)
    prev_h = prev_h.unsqueeze(1) # (N, 1, H)

    scores = torch.bmm(prev_h, A_flat) # (N, 1, D_a * D_a)
    scores = scores / (H ** 0.5)

    attn_weights = torch.softmax(scores, dim=2) # (N, 1, D_a * D_a)
    attn = torch.bmm(A_flat, attn_weights.transpose(1, 2)) # (N, H, 1)
    attn = attn.squeeze(2)

    attn_weights = attn_weights.reshape(N, D_a, D_a) 
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return attn, attn_weights


class AttentionLSTM(nn.Module):
    """
    This is our single-layer, uni-directional Attention module.

    Args:
    - input_dim: Input size, denoted as D before
    - hidden_dim: Hidden size, denoted as H before
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - Wattn: Weights for attention-to-hidden connections, of shape (H, 4H)
        - b: Biases, of shape (4H,)
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.Wattn = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self,
        x: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
        attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)
        - prev_c: The previous cell state, of shape (N, H)
        - attn: The attention embedding, of shape (N, H)

        Returns:
        - next_h: The next hidden state, of shape (N, H)
        - next_c: The next cell state, of shape (N, H)
        """
        next_h, next_c = None, None
        ######################################################################
        # TODO: Implement the forward pass for a single timestep of an       #
        # attention LSTM, which should be very similar to LSTM.step_forward  #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        H = prev_h.shape[1]

        a = x @ self.Wx + prev_h @ self.Wh + attn @ self.Wattn + self.b # (N, 4H)

        ai, af, ao, ag = torch.chunk(a, 4, dim=1)

        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return next_h, next_c

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM
        uses a hidden size of H, and we work over a minibatch containing
        N sequences. After running the LSTM forward, we return hidden states
        for all timesteps.

        Note that the initial cell state is passed as input, but the initial
        cell state is set to zero. Also note that the cell state is not
        returned; it is an internal variable to the LSTM and is not accessed
        from outside.

        h0 and c0 are same initialized as global image feature (avgpooled A)
        For simplicity, we implement scaled dot-product attention, which means
        in Eq. 4 of the paper (https://arxiv.org/pdf/1502.03044.pdf),
        f_{att}(a_i, h_{t-1}) equals to the scaled dot product of
        a_i and h_{t-1}.

        Args:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - A: The projected CNN feature activation, of shape (N, H, D_a, D_a)

        Returns:
        - hn: The hidden state output
        """

        # The initial hidden state h0 and cell state c0 are initialized
        # differently in AttentionLSTM from the original LSTM and hence
        # we provided them for you.
        h0 = A.mean(dim=(2, 3))  # Initial hidden state, of shape (N, H)
        c0 = h0  # Initial cell state, of shape (N, H)
        hn = None
        ######################################################################
        # TODO: Implement the forward pass for an attention LSTM running on  #
        # a sequence of input data, which should be very similar to          #
        # LSTM.forward                                                       #
        # Hint: dot_product_attention                                        #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        N, T, D = x.shape
        H = A.shape[1]
        D_a = A.shape[2]

        hn = torch.zeros(N, T, H, device=x.device, dtype=x.dtype)

        h = h0
        c = c0 

        for t in range(T):
            xt = x[:, t, :] # (N, D)
            attn_emb, _ = dot_product_attention(h, A) # (N, H), (N, D_a, D_a)
            h, c = self.step_forward(xt, h, c, attn_emb)
            hn[:, t, :] = h 
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return hn
