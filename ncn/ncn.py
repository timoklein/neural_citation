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
        x = torch.einsum("ijk -> ikj", x)
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
        x = [encoder(x) for encoder in self.encoder]

        # output shape: batch_size, list_length, num_filters
        x = torch.cat(x, dim=1).squeeze()

        # output shape: batch_size, list_length*num_filters
        x = x.view(self.bs, -1)

        # apply nonlinear mapping
        x = torch.tanh(self.fc(x))

        # output shape: batch_size, list_length, num_filters
        return x.view(-1, len(self.filter_list), self.num_filters)


# TODO: Fix this to work with model
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, hidden, encoder_outputs, mask):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        #mask = [batch size, src sent len]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src sent len, dec hid dim]
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src sent len, dec hid dim]
                
        energy = energy.permute(0, 2, 1)
        
        #energy = [batch size, dec hid dim, src sent len]
        
        #v = [dec hid dim]
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        
        #v = [batch size, 1, dec hid dim]
            
        attention = torch.bmm(v, energy).squeeze(1)
        
        #attention = [batch size, src sent len]
        
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)


# TODO: Fix this to work with model
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        #mask = [batch size, src sent len]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs, mask)
                
        #a = [batch size, src sent len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src sent len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [sent len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #sent len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        output = self.out(torch.cat((output, weighted, embedded), dim = 1))
        
        #output = [bsz, output dim]
        
        return output, hidden.squeeze(0), a.squeeze(1)


# TODO: Debug this
# TODO: Get this to work with batches
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
                       sos_idx: int,
                       eos_idx: int,
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
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

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
    
