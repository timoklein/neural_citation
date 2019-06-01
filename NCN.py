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
        super(NCN, self).__init__(config)
        self._use_authors = self.config.use_authors

        ##################################
        # Placeholders
        ##################################
        self.encoder_inputs = torch.Tensor(self.config.batch_size, max_encoder_length)
        self.decoder_inputs = torch.Tensor(self.config.batch_size, max_decoder_length)
        self.decoder_targets = torch.Tensor(self.config.batch_size, max_decoder_length)
        self.decoder_seq_len = torch.Tensor(self.config.batch_size)
        # self.dropout_keep_prob
        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)

        if self._use_authors:
            # assert self.config.author_num_filters == self.config.num_filters, "Currently, we require author num filters == num filters "
            max_author_len = self.config.max_author_len
            self.encoder_authors = torch.Tensor(self.config.batch_size, max_author_len)
            self.decoder_authors = torch.Tensor(self.config.batch_size, max_author_len)

            class Author():
                self._auth_embeddings = nn.Embedding(self.config.author_vocab_size, self.config.author_embed_size)
                self._auth_encoder_embed = self._auth_embeddings(self.encoder_authors)

        # context encoder

        # decoder

    def forward(self, x):
        pass
