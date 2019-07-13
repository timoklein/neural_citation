import logging
import random
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

import core
from core import Filters, DEVICE

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
        x = self.bn(F.relu(self.conv(x)))
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

        self.encoder = [TDNN(filter_size=f, embed_size = embed_size, num_filters=num_filters).to(DEVICE) 
                                for f in self.filter_list]
        self.fc = nn.Linear(self._num_filters_total, self._num_filters_total)
        self.bn = nn.BatchNorm1d(self._num_filters_total)

    def forward(self, x):
        """
        ## Input:  

        - **Tensor** *(batch size, embedding dimensions, sequence length)*:
            Input sequence.  

        ## Output:  

        - **Tensor** *(number of filter sizes, batch size, # filters)*:
            Output sequence.
        """
        x = [encoder(x) for encoder in self.encoder]

        # output shape: batch_size, list_length, num_filters
        x = torch.cat(x, dim=1).squeeze()

        # output shape: batch_size, list_length*num_filters
        x = x.view(self.bs, -1)
        logger.debug(f"x shape: {x.shape}")

        # apply nonlinear mapping
        x = self.bn(torch.tanh(self.fc(x)))

        # output shape: list_length, batch_size, num_filters
        return x.view(len(self.filter_list), -1, self.num_filters)


# TODO: Document this
class NCNEncoder(nn.Module):
    """
    Insert your description here.  
    
    ## Parameters:  
    - **context_filters** *(int)*: List of ints representing the context filter lengths.  
    - **author_filters** *(int)*: List of ints representing the author filter lengths.  
    - **context_vocab_size** *(int)*: Size of the context vocabulary. Used to train context embeddings.  
    - **title_vocab_size** *(int)*: Size of the title vocabulary. Used to train title embeddings.  
    - **author_vocab_size** *(int)*: Size of the author vocabulary. Used to train author embeddings.  
    - **num_filters** *(int)*: Number of filters applied in the TDNN layers of the model.   
    - **embed_size** *(int)*: Dimension of the learned author, context and title embeddings.  
    - **pad_idx** *(int)*: Index of the pad token in the vocabulary. Is set to zeros by the embedding layer.   
    - **batch_size** *(int)*: Training batch size.  
    - **dropout_p** *(float)*: Dropout probability for the dropout regularization layers.  
    - **authors** *(bool)*: Use author information in the encoder.   
    """
    def __init__(self, context_filters: Filters,
                       author_filters: Filters,
                       context_vocab_size: int,
                       author_vocab_size: int,
                       num_filters: int,
                       embed_size: int,
                       pad_idx: int,
                       batch_size: int,
                       dropout_p: float,
                       authors: bool):
        super().__init__()

        self.use_authors = authors

        self.dropout = nn.Dropout(dropout_p)

        # context encoder
        self.context_embedding = nn.Embedding(context_vocab_size, embed_size, padding_idx=pad_idx)
        self.context_encoder = TDNNEncoder(context_filters, num_filters, embed_size, batch_size)

        # author encoder
        if self.use_authors:
            self.author_embedding = nn.Embedding(author_vocab_size, embed_size, padding_idx=pad_idx)

            self.citing_author_encoder = TDNNEncoder(author_filters, num_filters, embed_size, batch_size)
            self.cited_author_encoder = TDNNEncoder(author_filters, num_filters, embed_size, batch_size)

    def forward(self, context, authors_citing=None, authors_cited=None):
        """
        ## Input:  
        
        - **Input 1** *(shapes)*:  
        
        ## Output:  
        
        - **Output 1** *(shapes)*: 
        """
        # Embed and encode context
        context = self.dropout(self.context_embedding(context))
        context = self.context_encoder(context)
        logger.debug(f"Context encoding shape: {context.shape}")

        if self.use_authors and authors_citing is not None and authors_cited is not None:
            logger.debug("Forward pass uses author information.")

            # Embed authors in shared space
            authors_citing = self.dropout(self.author_embedding(authors_citing))
            authors_cited = self.dropout(self.author_embedding(authors_cited))

            # Encode author information and concatenate
            authors_citing = self.citing_author_encoder(authors_citing)
            authors_cited = self.cited_author_encoder(authors_cited)
            logger.debug(f"Citing author encoding shape: {authors_citing.shape}")
            logger.debug(f"Cited author encoding shape: {authors_cited.shape}")

            # [N: batch_size, F: total # of filters (authors, cntxt), D: embedding size]
            return torch.cat([context, authors_citing, authors_cited], dim=0)
        
        return context


class Attention(nn.Module):
    """
    Base attention module as published in the paper https://arxiv.org/abs/1409.0473.
    The code is based on https://github.com/bentrevett/pytorch-seq2seq.  
    
    ## Parameters:  
    
    - **enc_num_filters** *(int)*: Number of filters used in the encoder.  
    - **dec_hid_dim** *(int)*: Dimensions of the decoder RNN layer hidden state.   
    """
    def __init__(self, enc_num_filters: int , dec_hid_dim: int):
        super().__init__()
        
        self.enc_num_filters = enc_num_filters
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear(enc_num_filters + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
    
    def forward(self, hidden, encoder_outputs):
        """
        ## Input:  
        
        - **hidden** *(batch_size, dec_hidden_dim)*: Hidden state of the decoder recurrent layer.  
        - **encoder_otuputs** *(number of filter sizes, batch size, # filters)*: 
            Encoded context and author information.  
        
        ## Output:  
        
        - **a** *(batch_size, number of filter sizes)*: 
            Tensor containing the attention weights for the encoded source data.
        """
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        logger.debug(f"Attention Batch size: {batch_size}")
        logger.debug(f"Attention weights: {src_len}")
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)    
        attention = torch.bmm(v, energy).squeeze(1)
        
        return torch.softmax(attention, dim=1)



class Decoder(nn.Module):
    """
    Attention decoder for a Seq2Seq model. Uses a GRU layer as recurrent unit.  
    The code is based on https://github.com/bentrevett/pytorch-seq2seq.  
    
    ## Parameters:  
    
    - **title_vocab_size** *(int)*: Size of the title vocabulary used in the embedding layer.  
    - **embed_size** *(int)*: Dimensions of the learned embeddings.  
    - **enc_num_filters *(int)*: Number of filters used in the TDNN convolution layer.  
    - **hidden_size** *(int)*: Specifies the dimensions of the hidden GRU layer state.  
    - **pad_idx** *(int)*: Index used for pad tokens. Will be ignored by the embedding layer.  
    - **dropout_p** *(float)*: Dropout probability.  
    - **attention** *(nn.Module)*: Module for computing the attention weights.  
    """
    def __init__(self, title_vocab_size: int, embed_size: int, enc_num_filters: int, hidden_size: int,
                 pad_idx: int, dropout_p: float, attention: nn.Module):
        super().__init__()

        self.embed_size = embed_size
        self.enc_num_filtes = enc_num_filters
        self.hidden_size = hidden_size
        self.title_vocab_size = title_vocab_size
        self.dropout_p = dropout_p
        self.attention = attention
        
        self.embedding = nn.Embedding(title_vocab_size, embed_size, padding_idx=pad_idx)
        
        self.rnn = nn.GRU(enc_num_filters + embed_size, hidden_size)
        
        self.out = nn.Linear(enc_num_filters*2 + embed_size, title_vocab_size)
        
        self.dropout = nn.Dropout(dropout_p)
    
    
    def init_hidden(self, bs: int):
        """Initializes the GRU hidden state to a tensor of zeros of appropriate size."""
        return torch.zeros(bs, self.hidden_size, device=DEVICE)
    
    def forward(self, title, hidden, encoder_outputs):
        """
        ## Input:  
        
        - **title** *(batch size)*: Batch of initial title tokens.  
        - **hidden** *(batch size, hidden_dim): Hidden state of the GRU unit.  
        - **encoder_otuputs** *(number of filter sizes, batch size, # filters)*: 
            Encoded context and author information. 
        
        ## Output:  
        
        - **output** *(batch size, vocab_size)*: Scores for each word in the vocab.  
        - **hidden** *(batch size, hidden_dim): Hidden state of the GRU unit.  
        """
        
        input = title.unsqueeze(0)
        logger.debug(f"Title shape: {title.shape}")
        logger.debug(f"Hidden shape: {hidden.shape}")
        logger.debug(f"Encoder output shape: {encoder_outputs.shape}")
        
        embedded = self.dropout(self.embedding(input))
        logger.debug(f"Embedded shape: {embedded.shape}")
         
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        logger.debug(f"Attention output shape: {a.shape}")
        logger.debug(f"Encoder outputs: {encoder_outputs.shape}")
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        logger.debug(f"Weighted shape: {weighted.shape}")
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        logger.debug(f"RNN input shape: {rnn_input.shape}")  
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        # sanity check
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        logger.debug(f"Hidden shape: {hidden.shape}")
        logger.debug(f"Output shape: {output.shape}")
        logger.debug(f"Weighted shape: {weighted.shape}")
        
        output = self.out(torch.cat((output, weighted, embedded), dim = 1))
        
        return output, hidden.squeeze(0)


# TODO: Create inference pass for NCN
class NeuralCitationNetwork(nn.Module):
    """
    PyTorch implementation of the neural citation network by Ebesu & Fang.  
    The original paper can be found here:  
    http://www.cse.scu.edu/~yfang/NCN.pdf.   
    The author's tensorflow code is on github:  
    https://github.com/tebesu/NeuralCitationNetwork.  

    ## Parameters:  
    - **context_filters** *(int)*: List of ints representing the context filter lengths.  
    - **author_filters** *(int)*: List of ints representing the author filter lengths.  
    - **context_vocab_size** *(int)*: Size of the context vocabulary. Used to train context embeddings.  
    - **title_vocab_size** *(int)*: Size of the title vocabulary. Used to train title embeddings.  
    - **author_vocab_size** *(int)*: Size of the author vocabulary. Used to train author embeddings.  
    - **pad_idx** *(int)*: Index of the pad token in the vocabulary. Is set to zeros by the embedding layer.   
    - **num_filters** *(int=128)*: Number of filters applied in the TDNN layers of the model.   
    - **authors** *(bool=True)*: Use author information in the encoder.  
    - **embed_size** *(int=128)*: Dimension of the learned author, context and title embeddings.  
    - **num_layers** *(int=2)*: Number of GRU layers.  
    - **hidden_size** *(int=128)*: Dimension of the GRU hidden states.  
    - **batch_size** *(int=32)*: Training batch size.  
    - **dropout_p** *(float=0.2)*: Dropout probability for the dropout regularization layers.  
    """
    def __init__(self, context_filters: Filters,
                       author_filters: Filters,
                       context_vocab_size: int,
                       title_vocab_size: int,
                       author_vocab_size: int,
                       pad_idx: int,
                       num_filters: int = 128,
                       authors: bool = True, 
                       embed_size: int = 128,
                       num_layers: int = 2,
                       hidden_size: int = 128,
                       batch_size: int = 32,
                       dropout_p: float = 0.2):
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

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bs = batch_size
        self._batched = self.bs > 1
        self.dropout_p = dropout_p

        # sanity check
        msg = (f"# Filters={self.num_filters}, Hidden dimension={self.hidden_size}, Embedding dimension={self.embed_size}"
               f"\nThese don't match!")
        assert self.num_filters == self.hidden_size == self.embed_size, msg

        #---------------------------------------------------------------------------------------------------------------
        # NCN MODEL
        
        # Encoder
        self.encoder = NCNEncoder(context_filters = self.context_filter_list,
                                  author_filters = self.author_filter_list,
                                  context_vocab_size = self.context_vocab_size,
                                  author_vocab_size = self.author_vocab_size,
                                  num_filters = self.num_filters,
                                  embed_size = self.embed_size,
                                  pad_idx = self.pad_idx,
                                  batch_size = self.bs,
                                  dropout_p= self.dropout_p,
                                  authors = self.use_authors)

        # attention decoder
        self.attention = Attention(self.num_filters , self.hidden_size)
        self.decoder = Decoder(title_vocab_size = self.title_vocab_size,
                               embed_size = self.embed_size,
                               enc_num_filters = self.num_filters,
                               hidden_size = self.hidden_size,
                               pad_idx = self.pad_idx,
                               dropout_p = self.dropout_p,
                               attention = self.attention)
        

        settings = (f"INITIALIZING NEURAL CITATION NETWORK WITH AUTHORS = {self.use_authors}"
                    f"\nRunning on: {DEVICE}"
                    f"\nNumber of model parameters: {self.count_parameters():,}"
                    f"\nEncoders: # Filters = {self.num_filters}, "
                        f"Context filter length = {self.context_filter_list},  Context filter length = {self.author_filter_list}"
                    f"\nEmbeddings: Dimension = {self.embed_size}, Pad index = {self.pad_idx}, Context vocab = {self.context_vocab_size}, "
                        f"Author vocab = {self.author_vocab_size}, Title vocab = {self.title_vocab_size}"
                    f"\nDecoder: # GRU cells = {self.num_layers}, Hidden size = {self.hidden_size}"
                    f"\nParameters: Batch size = {self.bs}, Dropout = {self.dropout_p}"
                    "\n--------------------------")
        
        logger.info(settings)

    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, context, title, authors_citing=None, authors_cited=None,
               teacher_forcing_ratio=1):
        """
        ## Inputs:  
    
        - **Tensor** *(N: batch size, D: embedding dimensions, L: sequence length)*:  
            Encoder input sequence.  
        
        ## Output:  
        
        - **Output 1**: *(shapes)* 
        """
        
        encoder_outputs = self.encoder(context, authors_citing, authors_cited)
        
        # maximum title sequence length
        max_len = title.shape[0]
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, self.bs, self.title_vocab_size).to(DEVICE)
                
        #first input to the decoder is the <sos> tokens
        output = title[0,:]
        
        hidden= self.decoder.init_hidden(self.bs)
        
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (title[t] if teacher_force else top1)

        return outputs

