import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms as T

class TDNN(nn.Module):
    """
    Single TDNN Block for the neural citation network.
    Consists of the following layers: Convolution, Batchnorm, ReLu, MaxPool.
    """
    def __init__(self, filter_size: int, num_filters: int = 64,):
        super().__init__()
        self.conv = nn.Conv2d()
        self.bn = nn.BatchNorm2d()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d()

    def forward(self, x):
        pass


class AttentionDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class NCN(nn.Module):
    def __init__(self, authors: bool = False, embed_size: int = 300):
        super().__init__()

        # context encoder

        # decoder

    def forward(self, x):
        pass