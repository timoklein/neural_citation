import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from typing import List

Filters = List[int]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device to GPU if available."""

MAX_LENGTH = 20

class TDNN(nn.Module):
    """
    Single TDNN Block for the neural citation network.  
    Implementation is based on:  
    https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf.  
    Consists of the following layers (in order): Convolution, Batchnorm, ReLu, MaxPool.   
    **Parameters**:   
    - *filter_size* (int): filter length for the convolutional operation  
    - *embed_size* (int): Dimension of the input word embeddings  
    - *num_filters* (int=64): Number of convolutional filters  
    **Input**:  
    - Tensor of shape: [N: batch size, D: embedding dimensions, L: sequence length].  
    **Output**:  
    - Tensor of shape: [batch_size, num_filters]. 
    """

    def __init__(self, filter_size: int, embed_size: int, num_filters: int = 64):
        super().__init__()
        # model input shape: [N: batch size, D: embedding dimensions, L: sequence length]
        self.conv = nn.Conv2d(1, num_filters, kernel_size=(embed_size, filter_size))
        self.bn = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        """
        Forward pass.
        """
        # output shape: [N: batch size, 1: channels, D: embedding dimensions, L: sequence length]
        x = x.unsqueeze(1)


        # output shape: batch_size, num_filters, 1, f(seq length)
        x = F.relu(self.bn(self.conv(x)))
        pool_size = x.shape[-1]

        # output shape: batch_size, num_filters, 1, 1
        x = F.max_pool2d(x, kernel_size=pool_size)

        # output shape: batch_size, 1, num_filters, 1
        return torch.einsum("nchw -> nhcw", x)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.2, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)



class NCN(nn.Module):
    """
    PyTorch implementation of the neural citation network by Ebesu & Fang.  
    The original paper can be found here:  
    http://www.cse.scu.edu/~yfang/NCN.pdf.   
    The author's tensorflow code is on github:  
    https://github.com/tebesu/NeuralCitationNetwork.  


    """
    def __init__(self, filters: Filters,
                       num_filters: int = 64,
                       authors: bool = False, 
                       w_embed_size: int = 300,
                       num_layers: int = 1,
                       hidden_dims: int = 64,
                       batch_size: int = 32):
        super().__init__()
        self._use_authors = authors
        self._filter_list = filters
        self._num_filters = num_filters
        self._bs = 32

        self._num_filters_total = len(filters)*num_filters
        
        # context encoder
        self.convs = [TDNN(filter_size=f, embed_size = w_embed_size, num_filters=num_filters) 
                      for f in self._filter_list]
        
        # Are inputs and outputs here really right?
        self.fc = nn.Linear(self._num_filters_total, self._num_filters_total)


        # decoder

    def forward(self, x):
        # encoder
        # output: List of tensors w. shape: batch size, 1, num_filters, 1
        x = [encoder(x) for encoder in self.convs]
        # output shape: batch_size, list_length, num_filters
        x = torch.cat(x, dim=1).squeeze()
        # output shape: batch_size, list_length*num_filters
        x = x.view(self._bs, -1)

        # apply nonlinear mapping
        x = torch.tanh(self.fc(x))
        x = x.view(-1, len(self._filter_list), self._num_filters)

        #------------------------------------------------------------------
        # decode

        return x
