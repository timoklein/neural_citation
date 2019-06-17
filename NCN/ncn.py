import torch
from torch import nn
import torch.nn.functional as F
from typing import List
import logging

Filters = List[int]
"""Custom data type representing a list of filter lengths."""

MAX_LENGTH = 20

class TDNN(nn.Module):
    """
    Single TDNN Block for the neural citation network.
    Implementation is based on:  
    https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf.  
    Consists of the following layers (in order): Convolution, Batchnorm, ReLu, MaxPool.  

    ## Parameters:   

    - **filter_size** *(int)*: filter length for the convolutional operation  
    - **embed_size** *(int)*: Dimension of the input word embeddings  
    - **num_filters** *(int=64)*: Number of convolutional filters  
    """

    def __init__(self, filter_size: int, embed_size: int, num_filters: int = 64):
        super().__init__()
        # model input shape: [N: batch size, D: embedding dimensions, L: sequence length]
        self.conv = nn.Conv2d(1, num_filters, kernel_size=(embed_size, filter_size))
        self.bn = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        """
        ## Input:  

        - **Tensor** *(N: batch size, D: embedding dimensions, L: sequence length)*:  
            Input sequence.

        ## Output:  

        - **Tensor** *(batch_size, num_filters)*:  
            Output sequence. 
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


class TDNNEncoder(nn.Module):
    """
    Encoder Module based on the TDNN architecture.
    Applies as list of filters with different region sizes on an input sequence.  
    
    ## Parameters:  
    
    - **filters** *(Filters)*: List of integers determining the filter lengths.    
    - **num_filters** *(int)*: Number of filters applied in the TDNN convolutional layers.  
    - **embed_size** *(int)*: Dimensions of the used embeddings.  
    - **bach_size** *(int)*: Training batch size. 
    """
    def __init__(self, filters: Filters,
                       num_filters: int,
                       embed_size: int,
                       batch_size: int):

        super().__init__()
        self.filter_list = filters
        self.num_filters = num_filters
        self.bs = batch_size
        self._num_filters_total = len(filters)*num_filters

        self.encoder = [TDNN(filter_size=f, embed_size = embed_size, num_filters=num_filters) 
                                for f in self.filter_list]
        self.fc = nn.Linear(self._num_filters_total, self._num_filters_total)

    def forward(self, x):
        """
        ## Input:  

        - **Tensor** *(N: batch size, D: embedding dimensions, L: sequence length)*:
            Input sequence.  

        ## Output:  

        - **Tensor** *(batch_size, number of filter sizes, num_filters)*:
            Output sequence.
        """
        # output: List of tensors w. shape: batch size, 1, num_filters, 1
        x = [encoder(x) for encoder in self.encoder]

        # output shape: batch_size, list_length, num_filters
        x = torch.cat(x, dim=1).squeeze()

        # output shape: batch_size, list_length*num_filters
        x = x.view(self.bs, -1)

        # apply nonlinear mapping
        x = torch.tanh(self.fc(x))

        # output shape: batch_size, list_length, num_filters
        return x.view(-1, len(self.filter_list), self.num_filters)



# Why do we nee an encoder and decoder Embedding?
# Because even though we use English as language for input and output,
# the words used are in the contexts and the cited paper's titles.
# This is especially pronounced when using a small vocabulary (like 20k words).
# TODO: Check how we can get only the last relevant output


class AttnDecoderRNN(nn.Module):
    """
    Decoder module for a seq2seq model. The implementation is based on the PyTorch documentation.
    The original code can be found here:  
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html.  
    Background: https://arxiv.org/pdf/1409.0473.pdf.  
    
    ## Parameters:  
    
    - **hidden_size** *(int)*: Dimensions of GRU's hidden state.  
    - **output_size** *(int)*: Output dimensions of the last linear layer (vocab_size)  
    - **dropout_p** *(float=0.2)*: Probability for dropout regularization. If 0, no regularization is applied.
        Dropout is also applied to the recurrent layers.  
    - **max_length** *(int)*: Maximum sequence length of the input (# of attention weights).   
    """
    def __init__(self, embed_size: int, 
                       vocab_size: int, 
                       dropout_p: int = 0.2, 
                       layers: int = 1,
                       max_length: int = MAX_LENGTH):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # embed input sequence, in our case cited paper title
        # self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        # always calculate 20 attention weights (don't use if input is shorter)
        # Why embed_size*2? We get hidden state and prev attention output as new input
        self.attn = nn.Linear(self.embed_size * 2, max_length)
        # Combine attention and hidden state
        self.attn_combine = nn.Linear(self.embed_size * 2, self.embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, dropout= dropout_p, num_layers=layers)

        # output
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        ## Inputs:  
    
        - **Input 1** *(shapes)*:   .    
        
        ## Outputs:  
        
        - **Output 1** *(shapes)*:   .    
        """
        # Embed input word index!!!! and apply dropout
        # Output is processed word for word
        # embedded = self.embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)


        attn_weights = torch.softmax(
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
        return torch.zeros(1, 1, self.hidden_size, device=self._device)



class NCN(nn.Module):
    """
    PyTorch implementation of the neural citation network by Ebesu & Fang.  
    The original paper can be found here:  
    http://www.cse.scu.edu/~yfang/NCN.pdf.   
    The author's tensorflow code is on github:  
    https://github.com/tebesu/NeuralCitationNetwork.  

    ## Parameters:  
    
    - **num_filters** *(int=64)*: Number of filters applied in the TDNN layers of the model.  
    - **authors** *(bool=False)*: Use additional author information or not.  
    - **w_emebd_size** *(int=300)*: Input word embedding dimensions.  
    - **num_layers** *(int=1)*: Number of RNN layers.  
    - **hidden_dims** *(int=64)*: Dimension of the RNN hidden states.  
    - **batch_size** *(int=32)*: Training batch size.  
    """
    def __init__(self, context_filters: Filters,
                       author_filters = Filters,
                       num_filters: int = 64,
                       authors: bool = False, 
                       embed_size: int = 300,
                       num_layers: int = 1,
                       hidden_dims: int = 64,
                       batch_size: int = 32):
        super().__init__()

        self.use_authors = authors
        self.context_filter_list = context_filters
        self.author_filter_list = author_filters
        self.num_filters = num_filters # num filters for context == num filters for authors
        self.bs = 32

        # ncn logging stuff
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # TODO: Train own context embeddings from dictionary

        # context encoder
        self.context_encoder = TDNNEncoder(self.context_filter_list, self.num_filters, embed_size, self.bs)

        if self.use_authors:
            # TODO: Train shared author embeddings from dictionary

            self.citing_author_encoder = TDNNEncoder(self.author_filter_list, self.num_filters, embed_size, self.bs)
            self.cited_author_encoder = TDNNEncoder(self.author_filter_list, self.num_filters, embed_size, self.bs)

        # TODO: Instantiate AttentionDecoder (Does this have to be different depending on authors?)

    def forward(self, context, title, authors_citing=None, authors_cited=None):
        """
        ## Inputs:  
    
        - **Tensor** *(N: batch size, D: embedding dimensions, L: sequence length)*:  
            Encoder input sequence.  
        
        ## Output:  
        
        - **Output 1**: *(shapes)* 
        """

        context = self.context_encoder(context)

        if self.use_authors and authors_citing is not None and authors_cited is not None:
            self.logger.info("Using Author information")

            authors_citing = self.citing_author_encoder(authors_citing)
            authors_cited = self.cited_author_encoder(authors_cited)
            cat_encodings = torch.cat([context, authors_citing, authors_cited], dim=1)
        


            


        return context # What does this thing actually return???? -> NLLLoss???
