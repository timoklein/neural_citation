import logging
import random
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

import ncn.core
from ncn.core import Filters, DEVICE

logger = logging.getLogger("neural_citation.ncn")


class TDNN(nn.Module):
    """
    Single TDNN Block for the neural citation network.
    Implementation is based on:  
    https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf.  
    Consists of the following layers (in order): Convolution, ReLu, Batchnorm, MaxPool.  

    ## Parameters:   

    - **filter_size** *(int)*: filter length for the convolutional operation  
    - **embed_size** *(int)*: Dimension of the input word embeddings  
    - **num_filters** *(int)*: Number of convolutional filters  
    """

    def __init__(self, filter_size: int, 
                       embed_size: int, 
                       num_filters: int):
        super().__init__()
        # model input shape: [N: batch size, D: embedding dimensions, L: sequence length]
        # no bias to avoid accumulating biases on padding
        self.conv = nn.Conv2d(1, num_filters, kernel_size=(embed_size, filter_size), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        ## Input:  

        - **Embedded sequence** *(batch size, seq length, embedding dimensions)*:  
            Tensor containing a batch of embedded input sequences.

        ## Output:  

        - **Convolved sequence** *(batch_size, num_filters)*:  
            Tensor containing the output. 
        """
        # [N: batch size, L: seq length, D embedding dimensions] -> [N: batch size, D embedding dimensions, L: seq length]
        x = x.permute(0, 2, 1)
        # output shape: [N: batch size, 1: channels, D: embedding dimensions, L: sequence length]
        x = x.unsqueeze(1)


        # output shape: batch_size, num_filters, 1, f(seq length)
        x = F.relu(self.conv(x))
        pool_size = x.shape[-1]

        # output shape: batch_size, num_filters, 1, 1
        x = F.max_pool2d(x, kernel_size=pool_size)

        # output shape: batch_size, 1, num_filters, 1
        return x.permute(0, 2, 1, 3)


class TDNNEncoder(nn.Module):
    """
    Encoder Module based on the TDNN architecture.
    Applies as list of filters with different region sizes on an input sequence.  
    The resulting feature maps are then allowed to interact with each other across a fully connected layer.  
    
    ## Parameters:  
    
    - **filters** *(Filters)*: List of integers determining the filter lengths.    
    - **num_filters** *(int)*: Number of filters applied in the TDNN convolutional layers.  
    - **embed_size** *(int)*: Dimensions of the used embeddings.  
    """
    def __init__(self, filters: Filters,
                       num_filters: int,
                       embed_size: int):

        super().__init__()
        self.filter_list = filters
        self.num_filters = num_filters
        self._num_filters_total = len(filters)*num_filters

        self.encoder = nn.ModuleList([TDNN(filter_size=f, embed_size = embed_size, num_filters=num_filters).to(DEVICE) 
                                        for f in self.filter_list])
        self.fc = nn.Linear(self._num_filters_total, self._num_filters_total)


    def forward(self, x: Tensor) -> Tensor:
        """
        ## Input:  

        - **Embeddings** *(batch size, seq length, embedding dimensions)*:
            Embedded input sequence.  

        ## Output:  

        - **Encodings** *(number of filter sizes, batch size, # filters)*:
            Tensor containing the complete context/author encodings.
        """
        x = [encoder(x) for encoder in self.encoder]
        assert len(set([e.shape[0] for e in x])) == 1, "Batch sizes don't match!"


        # output shape: batch_size, list_length, num_filters
        x = torch.cat(x, dim=1).squeeze(3)

        batch_size = x.shape[0]

        # output shape: batch_size, list_length*num_filters
        x = x.view(batch_size, -1)

        # apply nonlinear mapping
        x = torch.tanh(self.fc(x))

        # output shape: list_length, batch_size, num_filters
        return x.view(len(self.filter_list), -1, self.num_filters)



class NCNEncoder(nn.Module):
    """
    Encoder for the NCN model. Initializes TDNN Encoders for context and authors and concatenates the output.    
    
    ## Parameters:  
    - **context_filters** *(int)*: List of ints representing the context filter lengths.  
    - **author_filters** *(int)*: List of ints representing the author filter lengths.  
    - **context_vocab_size** *(int)*: Size of the context vocabulary. Used to train context embeddings.  
    - **title_vocab_size** *(int)*: Size of the title vocabulary. Used to train title embeddings.  
    - **author_vocab_size** *(int)*: Size of the author vocabulary. Used to train author embeddings.  
    - **num_filters** *(int)*: Number of filters applied in the TDNN layers of the model.   
    - **embed_size** *(int)*: Dimension of the learned author, context and title embeddings.  
    - **pad_idx** *(int)*: Index of the pad token in the vocabulary. Is set to zeros by the embedding layer.   
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
                       dropout_p: float,
                       authors: bool):
        super().__init__()

        self.use_authors = authors

        self.dropout = nn.Dropout(dropout_p)

        # context encoder
        self.context_embedding = nn.Embedding(context_vocab_size, embed_size, padding_idx=pad_idx)
        self.context_encoder = TDNNEncoder(context_filters, num_filters, embed_size)

        # author encoder
        if self.use_authors:
            self.author_embedding = nn.Embedding(author_vocab_size, embed_size, padding_idx=pad_idx)

            self.citing_author_encoder = TDNNEncoder(author_filters, num_filters, embed_size)
            self.cited_author_encoder = TDNNEncoder(author_filters, num_filters, embed_size)

    def forward(self, context: Tensor, 
                authors_citing: Tensor = None, authors_cited: Tensor = None) -> Tensor:
        """
        ## Input:  
        
        - **context** *(batch size, seq_length)*: 
            Tensor containing a batch of context indices.  
        - **authors_citing=None** *(batch size, seq_length)*:
            Tensor containing a batch of citing author indices.  
        - **authors_cited=None** *(batch size, seq_length)*: 
            Tensor containing a batch of cited author indices.
        
        ## Output:  
        
        - **output** *(batch_size, total # of filters (authors, cntxt), embedding size)*: 
            If authors= True the output tensor contains the concatenated context and author encodings.
            Else the encoded context is returned.
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
    Bahndanau attention module as published in the paper https://arxiv.org/abs/1409.0473.
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
    
    def forward(self, hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
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
        
        # Use only the last hidden state in case of multiple layers, i.e. hidden[-1]
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
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
    - **num_layers** *(float)*: Number of GRU layers.  
    - **attention** *(nn.Module)*: Module for computing the attention weights.  
    - **show_attention** *(bool)*: If True, the decoder also returns the attention weight matrix.  
    """
    def __init__(self, title_vocab_size: int, embed_size: int, enc_num_filters: int, hidden_size: int,
                 pad_idx: int, dropout_p: float, num_layers: int,
                 attention: nn.Module, show_attention: bool):
        super().__init__()

        self.num_layers = num_layers
        self.embed_size = embed_size
        self.enc_num_filtes = enc_num_filters
        self.hidden_size = hidden_size
        self.title_vocab_size = title_vocab_size
        self.dropout_p = dropout_p
        self.attention = attention
        self.show_attention = show_attention
        
        self.embedding = nn.Embedding(title_vocab_size, embed_size, padding_idx=pad_idx)
        self.rnn = nn.GRU(input_size=enc_num_filters + embed_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=self.dropout_p)
        
        self.out = nn.Linear(enc_num_filters*2 + embed_size, title_vocab_size)
        
        self.dropout = nn.Dropout(dropout_p)
    
    
    def init_hidden(self, bs: int):
        """Initializes the RNN hidden state to a tensor of zeros of appropriate size."""
        return torch.zeros(self.num_layers, bs, self.hidden_size, device=DEVICE)
    
    def forward(self, title: Tensor, hidden: Tensor, encoder_outputs: Tensor) -> Tuple[Tensor, ...]:
        """
        ## Input:  
        
        - **title** *(batch size)*: Batch of initial title tokens.  
        - **hidden** *(batch size, hidden_dim): Hidden state of the recurrent unit.  
        - **encoder_otuputs** *(number of filter sizes, batch size, # filters)*: 
            Encoded context and author information. 
        
        ## Output:  
        
        - **output** *(batch size, vocab_size)*: Scores for each word in the vocab.  
        - **hidden** *(batch size, hidden_dim): Hidden state of the recurrent unit.  
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
        output, hidden = self.rnn(rnn_input, hidden)
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        logger.debug(f"Hidden shape: {hidden.shape}")
        logger.debug(f"Decoder Output shape: {output.shape}")
        logger.debug(f"Weighted shape: {weighted.shape}")
        
        output = self.out(torch.cat((output, weighted, embedded), dim = 1))
        
        if self.show_attention:
            return output, hidden, a.squeeze(1)

        return output, hidden



class NeuralCitationNetwork(nn.Module):
    """
    PyTorch implementation of the neural citation network by Ebesu & Fang.  
    The original paper can be found here:  
    http://www.cse.scu.edu/~yfang/NCN.pdf.   
    The author's tensorflow code is on github:  
    https://github.com/tebesu/NeuralCitationNetwork.  

    ## Parameters:  
    - **context_filters** *(Filters)*: List of ints representing the context filter lengths.  
    - **author_filters** *(Filters)*: List of ints representing the author filter lengths.  
    - **context_vocab_size** *(int)*: Size of the context vocabulary. Used to train context embeddings.  
    - **title_vocab_size** *(int)*: Size of the title vocabulary. Used to train title embeddings.  
    - **author_vocab_size** *(int)*: Size of the author vocabulary. Used to train author embeddings.  
    - **pad_idx** *(int)*: Index of the pad token in the vocabulary. Is set to zeros by the embedding layer.   
    - **num_filters** *(int)*: Number of filters applied in the TDNN layers of the model.   
    - **authors** *(bool)*: Use author information in the encoder.  
    - **embed_size** *(int)*: Dimension of the learned author, context and title embeddings.  
    - **num_layers** *(int)*: Number of recurrent layers.  
    - **hidden_size** *(int)*: Dimension of the recurrent unit hidden states.  
    - **dropout_p** *(float=0.2)*: Dropout probability for the dropout regularization layers.  
    - **show_attention** *(bool=false)*: Returns attention tensors if true.  
    """
    def __init__(self, context_filters: Filters,
                       author_filters: Filters,
                       context_vocab_size: int,
                       title_vocab_size: int,
                       author_vocab_size: int,
                       pad_idx: int,
                       num_filters: int,
                       authors: bool, 
                       embed_size: int,
                       num_layers: int, 
                       hidden_size: int,
                       dropout_p: float = 0.2,
                       show_attention: bool = False):
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

        self.dropout_p = dropout_p
        self.show_attention = show_attention

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
                               num_layers = self.num_layers,
                               attention = self.attention,
                               show_attention=self.show_attention)
        

        self.settings = (
            f"INITIALIZING NEURAL CITATION NETWORK WITH AUTHORS = {self.use_authors}"
            f"\nRunning on: {DEVICE}"
            f"\nNumber of model parameters: {self.count_parameters():,}"
            f"\nEncoders: # Filters = {self.num_filters}, "
            f"Context filter length = {self.context_filter_list},  Context filter length = {self.author_filter_list}"
            f"\nEmbeddings: Dimension = {self.embed_size}, Pad index = {self.pad_idx}, Context vocab = {self.context_vocab_size}, "
            f"Author vocab = {self.author_vocab_size}, Title vocab = {self.title_vocab_size}"
            f"\nDecoder: # GRU cells = {self.num_layers}, Hidden size = {self.hidden_size}"
            f"\nParameters: Dropout = {self.dropout_p}, Show attention = {self.show_attention}"
            "\n-------------------------------------------------"
        )

    def count_parameters(self):
        """Calculates the number of trainable parameters.""" 
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, context: Tensor, title: Tensor, 
                authors_citing: Tensor = None, authors_cited: Tensor = None,
                teacher_forcing_ratio: float = 1):
        """
        ## Parameters:  

        - **teacher_forcing_ratio** *(float=1)*: Determines the ratio with which
            the model is fed the true output to predict the next token. Defaults to 1 which means
            a token is always conditioned on the true previous output.

        ## Inputs:  
    
        - **context** *(batch size, seq_length)*: 
            Tensor containing a batch of context indices.  
        - **title** *(seq_length, batch size)*: 
            Tensor containing a batch of title indices. Note: not batch first!
        - **authors_citing=None** *(batch size, seq_length):
            Tensor containing a batch of citing author indices.  
        - **authors_cited=None** *(batch size, seq_length)*: 
            Tensor containing a batch of cited author indices. 
        
        ## Output:  
        
        - **output** *(batch_size, seq_len, title_vocab_len)*: 
            Tensor containing the predictions of the decoder.
         **attentions** *(batch_size, title_vocab_len)*: 
            Tensor containing the decoder attention states.
        """
        
        encoder_outputs = self.encoder(context, authors_citing, authors_cited)
        
        batch_size = title.shape[1]
        max_len = title.shape[0]
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, self.title_vocab_size).to(DEVICE)   
        #first input to the decoder is the <sos> tokens
        output = title[0,:]

        if self.show_attention:
            attentions = torch.zeros((max_len, batch_size, encoder_outputs.shape[0])).to(DEVICE)
            logger.debug(f"Attentions viz shape: {attentions.shape}")
        
        hidden= self.decoder.init_hidden(batch_size)
        
        for t in range(1, max_len):
            if self.show_attention:
                output, hidden, attention = self.decoder(output, hidden, encoder_outputs)
                logger.debug(f"Attentions output shape: {attention.shape}")
                attentions[t] = attention
            else:
                output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (title[t] if teacher_force else top1)

        logger.debug(f"Model output shape: {outputs.shape}")

        if self.show_attention:
            return outputs, attentions
        
        return outputs