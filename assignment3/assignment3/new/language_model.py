import numpy as np
from segtok import tokenizer
import torch as th
from torch import nn


# Using a basic RNN/LSTM for Language modeling
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, rnn_size, num_layers=1, dropout=0):
        super().__init__()
        
        # Create an embedding layer of shape [vocab_size, rnn_size]
        # Use nn.Embedding
        # That will map each word in our vocab into a vector of rnn_size size.
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = rnn_size)

        # Create an LSTM layer of rnn_size size. Use any features you wish.
        # We will be using batch_first convention
        self.lstm = nn.LSTM(input_size = rnn_size, hidden_size = rnn_size, dropout = dropout, num_layers = num_layers, batch_first = True)
        #self.lstm = nn.LSTM(input_size = rnn_size, hidden_size = rnn_size, num_layers = num_layers, batch_first = True)
        # LSTM layer does not add dropout to the last hidden output.
        # Add this if you wish.
        #self.conv = nn.Conv1d(20, 20, 3, padding = 1, stride=1)
        #self.layernorm = 
        self.dropout = nn.Dropout(p=dropout)
        # Use a dense layer to project the outputs of the RNN cell into logits of
        # the size of vocabulary (vocab_size).
        self.output = nn.Linear(rnn_size, vocab_size)

        
    def forward(self,x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        #conv_out = self.conv(lstm_out)
        norm_out = nn.functional.layer_norm(lstm_out, (lstm_out.shape[1], lstm_out.shape[2]), weight=None, eps=1e-05)
        lstm_drop = self.dropout(norm_out)
        #logits = self.output(lstm_out)
        logits = self.output(lstm_drop)
        return logits


'''
from typing import Optional, List
from collections import namedtuple

import torch as th
from torch import nn
from torch.nn import functional as F


class EmbeddingTranspose(nn.Module):
    """Multiply by the transpose of an embedding layer
    """
    def __init__(self, embedding_layer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = embedding_layer

    def forward(self, inputs):
        embed_mat = self.embedding.weight.detach()
        return th.matmul(inputs, embed_mat.T)


class PositionEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super(PositionEmbedding, self).__init__()

        assert hidden_size % 2 == 0 and 'Model vector size must be even for sinusoidal encoding'
        power = th.arange(0, hidden_size, step=2, dtype=th.float32)[:] / hidden_size
        divisor = 10000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

    def forward(self, inputs, start=1):
        assert inputs.shape[-1] == self.hidden_size and 'Input final dim must match model hidden size'

        batch_size = inputs.shape[0]
        sequence_length = inputs.shape[1]

        seq_pos = th.arange(start, sequence_length + start, dtype=th.float32)
        seq_pos_expanded = seq_pos[None,:,None]
        index = seq_pos_expanded.repeat(*[1,1,self.hidden_size//2])
        
        sin_embedding = th.sin(index / self.divisor)
        cos_embedding = th.cos(index / self.divisor)
        
        position_shape = (1, sequence_length, self.hidden_size) # fill in the other two dimensions
        position_embedding = th.stack((sin_embedding,cos_embedding), dim=3).view(position_shape)
        pos_embed_deviced = position_embedding.to("cuda" if inputs.is_cuda else "cpu")

        return  inputs + pos_embed_deviced# add the embedding to the input


class TransformerFeedForward(nn.Module):
    def __init__(self, input_size,
                 filter_size,
                 hidden_size,
                 dropout):
        super(TransformerFeedForward, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.feed_forward = nn.Sequential(
                                nn.Linear(input_size,filter_size),
                                nn.ReLU(),
                                nn.Linear(filter_size,hidden_size)
                            )
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
        self.feed_forward.apply(weights_init)
        self.dropout = nn.Dropout(0 if dropout is None else dropout)

    def forward(self, inputs):      
        norm_input = self.norm(inputs)
        dense_out = self.feed_forward(norm_input)
        dense_drop =  self.dropout(dense_out)# Add the dropout here       
        return  dense_drop + inputs# Add the residual here





class LanguageModel(nn.Module):
    def __init__(self, vocab_size, rnn_size, num_layers=1, dropout=0):
        super().__init__()
        
        # Create an embedding layer of shape [vocab_size, rnn_size]
        # Use nn.Embedding
        # That will map each word in our vocab into a vector of rnn_size size.
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = rnn_size)
        self.posembedding = PositionEmbedding(hidden_size = rnn_size)

        self.norm1 = nn.LayerNorm(rnn_size)
        self.norm2 = nn.LayerNorm(rnn_size)
        self.norm3 = nn.LayerNorm(rnn_size)

        self.feed_forward1 = TransformerFeedForward(input_size = rnn_size, filter_size = rnn_size, hidden_size = rnn_size, dropout = dropout)
        self.feed_forward2 = TransformerFeedForward(input_size = rnn_size, filter_size = rnn_size, hidden_size = rnn_size, dropout = dropout)
        self.feed_forward3 = TransformerFeedForward(input_size = rnn_size, filter_size = rnn_size, hidden_size = rnn_size, dropout = dropout)
        # Create an LSTM layer of rnn_size size. Use any features you wish.
        # We will be using batch_first convention
        #self.lstm = nn.LSTM(input_size = rnn_size, hidden_size = rnn_size, dropout = dropout, num_layers = num_layers, batch_first = True)
        self.lstm1 = nn.LSTM(input_size = rnn_size, hidden_size = rnn_size, num_layers = num_layers, batch_first = True)
        self.lstm2 = nn.LSTM(input_size = rnn_size, hidden_size = rnn_size, num_layers = num_layers, batch_first = True)
        self.lstm = nn.LSTM(input_size = rnn_size, hidden_size = rnn_size, num_layers = num_layers, batch_first = True)
        # LSTM layer does not add dropout to the last hidden output.
        # Add this if you wish.
        # Use a dense layer to project the outputs of the RNN cell into logits of
        # the size of vocabulary (vocab_size).
        self.output = nn.Linear(rnn_size, vocab_size)

        
    def forward(self,x):
        embeds = self.embedding(x)
        posembeds = self.posembedding(embeds)

        norm1 = self.norm1(posembeds)
        lstm1, _ = self.lstm1(norm1)
        res1 =  lstm1 + posembeds
        forward1 = self.feed_forward1(res1)

        norm2 = self.norm2(forward1)
        lstm2, _ = self.lstm2(norm2)
        res2 = lstm2 + forward1
        forward2 = self.feed_forward2(res2)

        norm3 = self.norm3(forward2)
        lstm3, _ = self.lstm(norm3)
        res3 = lstm3 + forward2
        forward3 = self.feed_forward3(res3)

        logits = self.output(forward3)

        return logits
'''