import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms as T

from typing import Collection


class TDNN(nn.Module):
    """
    Single TDNN Block for the neural citation network.
    Consists of the following layers (in order): Convolution, Batchnorm, ReLu, MaxPool.
    Input is a tensor of shape: (batch_size, 1, embedding dimension, sequence length)
    Output is a tensor of shape: (batch_size, num_filters)
    * __filter size__(int):         filter length for the convolutional operation
    * __embed_size__(int):          Dimension of the input word embeddings
    * __num_filters__(int=64):      Number of filters to be applied in the convolution
    """

    def __init__(self, filter_size: int, embed_size: int, num_filters: int = 64):
        super().__init__()
        # input shape: N, C, D, L (we have only one input channel)
        self.conv = nn.Conv2d(1, num_filters, kernel_size=(embed_size, filter_size))
        self.bn = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        pool_size = x.shape[-1]
        return F.max_pool2d(x, kernel_size=pool_size).squeeze()


class AttentionDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class NCN(nn.Module):
    def __init__(self, config, max_encoder_length, max_decoder_length, authors: bool = False, embed_size: int = 300):
        super().__init__(config)
        self._use_authors = authors


        # context encoder

        # decoder

    def forward(self, x):
        pass
