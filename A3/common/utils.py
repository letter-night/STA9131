"""
General utilities to help with implementation
"""
import random

import matplotlib.pyplot as plt
import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure
    that our environment is correctly set up on Google Colab.
    """
    print('Hello from common!')


def reset_seed(number):
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)
    return


def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    Inputs:
    - tensor: A torch tensor of shape (3, H, W) with elements
              in the range [0, 1]

    Returns:
    - ndarr: A uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to('cpu', torch.uint8).numpy()
    return ndarr


def visualize_dataset(images, labels=None, nrow=12, classes=None, nchw=False):
    """
    Visualize torch.utils.data.Dataset class.
    Assume data and targets are instances of the input dataset.

    Inputs:
    - images: Tensor of images or torch.utils.data.Dataset class instance
    - labels: List of labels
    - nrow: Number of images per row.
    - classes: Class names to display. If given, i-th row shows i-th class.
      For CIFAR-10, e.g.,
      ['plane', 'car', 'bird', 'cat', 'deer',
       'dog', 'frog', 'horse', 'ship', 'truck']
    - nchw: True if images in the shape (N, C, H, W)
    """

    # Protected lazy import.
    from torchvision.utils import make_grid

    if isinstance(images, torch.utils.data.Dataset):
        dataset = images
        images = torch.tensor(dataset.data)
        labels = torch.tensor(dataset.targets)

    samples = []
    img_half_width = images.shape[2] // 2
    if classes:
        for y, cls in enumerate(classes):
            tx = -4
            ty = (img_half_width * 2 + 2) * y + (img_half_width + 2)
            plt.text(tx, ty, cls, ha='right')
            inds = (labels == y).nonzero().view(-1)
            ind = inds[torch.randperm(inds.shape[0])][:nrow]
            samples.append(images[ind])
        samples = torch.cat(samples, dim=0)
    else:
        nrow_sq = nrow * nrow
        ind = torch.randperm(images.shape[0])[:nrow_sq]
        samples = images[ind]
    if not nchw:  # make_grid gets NCHW
        samples = samples.permute(0, 3, 1, 2)
    img = make_grid(samples, nrow=nrow)

    if img.dtype == torch.uint8:
        plt.imshow(img.permute(1, 2, 0).to('cpu'))
    else:
        plt.imshow(tensor_to_image(img))
    plt.axis('off')
    plt.show()
