import torch
from torch import nn
import torch.nn.functional as F
from typing import List
import logging

from core import Filters, MAX_LENGTH
import logging_setup

logger = logging.getLogger("neural_citation.ncn")

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

    def __init__(self, filter_size: int, 
                       embed_size: int, 
                       num_filters: int = 64):
        super().__init__()
        # model input shape: [N: batch size, D: embedding dimensions, L: sequence length]
        # no bias to avoid accumulating biases on padding
        self.conv = nn.Conv2d(1, num_filters, kernel_size=(embed_size, filter_size), bias=False)
        self.bn = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        """
        ## Input:  

        - **Tensor** *(L: sequence length, N: batch size, D: embedding dimensions)*:  
            Input sequence.

        ## Output:  

        - **Tensor** *(batch_size, num_filters)*:  
            Output sequence. 
        """
        # [N: batch size, L: seq length, D embedding dimensions] -> [N: batch size, D embedding dimensions, L: seq length]
        x = x.permute(0, 2, 1)
        # output shape: [N: batch size, 1: channels, D: embedding dimensions, L: sequence length]
        x = x.unsqueeze(1)


        # output shape: batch_size, num_filters, 1, f(seq length)
        x = F.relu(self.bn(self.conv(x)))
        pool_size = x.shape[-1]

        # output shape: batch_size, num_filters, 1, 1
        x = F.max_pool2d(x, kernel_size=pool_size)

        # output shape: batch_size, 1, num_filters, 1
        return x.permute(0, 2, 1, 3)


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
        x = [encoder(x) for encoder in self.encoder]

        # output shape: batch_size, list_length, num_filters
        x = torch.cat(x, dim=1).squeeze()

        # output shape: batch_size, list_length*num_filters
        x = x.view(self.bs, -1)

        # apply nonlinear mapping
        x = torch.tanh(self.fc(x))

        # output shape: batch_size, list_length, num_filters
        return x.view(-1, len(self.filter_list), self.num_filters)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, hidden, encoder_outputs):
        
        # we start with hidden initialized to zero as our encoder has no hidden dim
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, num_filters, # filters]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # hidden = [batch size, src sent len, dec hid dim]
        # size is already fitting in our model
        # encoder_outputs = [batch size, src sent len, # filters]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        # energy = [batch size, src sent len, dec hid dim]
        energy = energy.permute(0, 2, 1)
        
        # energy = [batch size, dec hid dim, src sent len]
        
        # v = [dec hid dim]
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        
        # v = [batch size, 1, dec hid dim]
                
        attention = torch.bmm(v, energy).squeeze(1)
        
        # attention= [batch size, src len]
        
        return torch.softmax(attention, dim=1)


# TODO: Implement Decoder architecture




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
                       author_filters: Filters,
                       context_vocab_size: int,
                       title_vocab_size: int,
                       author_vocab_size: int,
                       pad_idx: int,
                       num_filters: int = 128,
                       authors: bool = False, 
                       embed_size: int = 128,
                       num_layers: int = 1,
                       hidden_dims: int = 128,
                       batch_size: int = 32,
                       dropout_p: float = 0.3):
        super().__init__()


        self.use_authors = authors
        self.context_filter_list = context_filters
        self.author_filter_list = author_filters
        self.num_filters = num_filters # num filters for context == num filters for authors

        self.embed_size = embed_size
        self.context_vocab_size = context_vocab_size
        self.title_vocab_size = title_vocab_size
        self.author_vocab_size = author_vocab_size
        self.pad_idx = pad_idx

        self.hidden_dims = hidden_dims
        self.num_layers = num_layers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bs = batch_size
        self._batched = self.bs > 1
        self.dropout_p = dropout_p


        # sanity check
        msg = (f"# Filters={self.num_filters}, Hidden dimension={self.hidden_dims}, Embedding dimension={self.embed_size}"
               f"\nThese don't match!")
        assert self.num_filters == self.hidden_dims == self.embed_size, msg

        #---------------------------------------------------------------------------------------------------------------
        # NCN MODEL
        self.dropout = nn.Dropout(self.dropout_p)

        # context encoder
        self.context_embedding = nn.Embedding(self.context_vocab_size, self.embed_size, padding_idx=self.pad_idx)
        self.context_encoder = TDNNEncoder(self.context_filter_list, self.num_filters, embed_size, self.bs)

        # author encoder
        if self.use_authors:
            self.author_embedding = nn.Embedding(self.author_vocab_size, self.embed_size, padding_idx=self.pad_idx)

            self.citing_author_encoder = TDNNEncoder(self.author_filter_list, self.num_filters, embed_size, self.bs)
            self.cited_author_encoder = TDNNEncoder(self.author_filter_list, self.num_filters, embed_size, self.bs)

        # decoder
        self.title_embedding = nn.Embedding(self.title_vocab_size, self.embed_size, padding_idx=self.pad_idx)

        # TODO: Instantiate Decoder


    def forward(self, context, title, hidden=None, authors_citing=None, authors_cited=None):
        """
        ## Inputs:  
    
        - **Tensor** *(N: batch size, D: embedding dimensions, L: sequence length)*:  
            Encoder input sequence.  
        
        ## Output:  
        
        - **Output 1**: *(shapes)* 
        """

        # Embed and encode context
        context = self.dropout(self.context_embedding(context))
        context = self.context_encoder(context)

        if self.use_authors and authors_citing is not None and authors_cited is not None:
            logger.info("Using Author information")

            # Embed authors in shared space
            authors_citing = self.dropout(self.author_embedding(authors_citing))
            authors_cited = self.dropout(self.author_embedding(authors_cited))

            # Encode author information and concatenate
            authors_citing = self.citing_author_encoder(authors_citing)
            authors_cited = self.cited_author_encoder(authors_cited)
            # [N: batch_size, F: total # of filters (authors, cntxt), D: embedding size]
            cat_encodings = torch.cat([context, authors_citing, authors_cited], dim=1)
        
        # Embed title
        title = self.dropout(self.title_embedding(title))
    
