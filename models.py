import math

import numpy as np
from torch import Tensor
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


class PositionalEncoding(nn.Module):
    """
    Classic Attention-is-all-you-need positional encoding.
    From PyTorch docs.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

def generate_square_subsequent_mask(size: int):
    """Generate a triangular (size, size) mask. From PyTorch docs."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Transformer(nn.Module):
    """
    Classic Transformer that both encodes and decodes.

    Prediction-time inference is done greedily.
    NOTE: start token is hard-coded to be 0, end token to be 1. If changing, update predict() accordingly.
    """

    def __init__(self, num_classes: int, max_output_length: int, dim: int = 128):
        super().__init__()

        # Parameters
        self.dim = dim
        self.max_output_length = max_output_length
        nhead = 4
        num_layers = 4
        dim_feedforward = dim

        # Encoder part
        self.embedding = nn.Embedding(num_classes, dim)
        self.pos_encoder = PositionalEncoding(d_model=self.dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )

        # Decoder part
        self.y_mask = generate_square_subsequent_mask(self.max_output_length)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=self.dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.fc = nn.Linear(self.dim, num_classes)

        # It is empirically important to initialize weights properly
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (B, C, Sy) logits
        """
        encoded_x = self.encode(x)  # (Sx, B, E)
        output = self.decode(y, encoded_x)  # (Sy, B, C)
        return output.permute(1, 2, 0)  # (B, C, Sy)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
        Output
            (Sx, B, E) embedding
        """
        x = x.permute(1, 0)  # (Sx, B, E)
        x = self.embedding(x) * math.sqrt(self.dim)  # (Sx, B, E)
        x = self.pos_encoder(x)  # (Sx, B, E)
        x = self.transformer_encoder(x)  # (Sx, B, E)
        return x

    def decode(self, y: torch.Tensor, encoded_x: torch.Tensor) -> torch.Tensor:
        """
        Input
            encoded_x: (Sx, B, E)
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (Sy, B, C) logits
        """
        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.dim)  # (Sy, B, E)
        y = self.pos_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)  # (Sy, Sy)
        output = self.transformer_decoder(y, encoded_x, y_mask)  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, C)
        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to use at inference time. Predict y from x one token at a time. This method is greedy
        decoding. Beam search can be used instead for a potential accuracy boost.
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
        Output
            (B, C, Sy) logits
        """
        encoded_x = self.encode(x)

        output_tokens = (torch.ones((x.shape[0], self.max_output_length))).type_as(x).long() # (B, max_length)
        output_tokens[:, 0] = 0  # Set start token
        for Sy in range(1, self.max_output_length):
            y = output_tokens[:, :Sy]  # (B, Sy)
            output = self.decode(y, encoded_x)  # (Sy, B, C)
            output = torch.argmax(output, dim=-1)  # (Sy, B)
            output_tokens[:, Sy] = output[-1:]  # Set the last output token
        return output_tokens



if __name__ == '__main__':
    N = 10000
    S = 32  # target sequence length. input sequence will be twice as long
    C = 128  # number of "classes", including 0, the "start token", and 1, the "end token"
    Y = (torch.rand((N * 10, S - 2)) * (C - 2)).long() + 2  # Only generate ints in (2, 99) range
    # Make sure we only have unique rows
    Y = torch.tensor(np.unique(Y, axis=0)[:N])
    X = torch.repeat_interleave(Y, 2, dim=1)
    # Add special 0 "start" and 1 "end" tokens to beginning and end
    Y = torch.cat([torch.zeros((N, 1)), Y, torch.ones((N, 1))], dim=1).long()
    X = torch.cat([torch.zeros((N, 1)), X, torch.ones((N, 1))], dim=1).long()
    # Look at the data
    print(X, X.shape)
    print(Y, Y.shape)
    print(Y.min(), Y.max())

    BATCH_SIZE = 128
    TRAIN_FRAC = 0.8

    dataset = list(zip(X, Y))  # This fulfills the pytorch.utils.data.Dataset interface

    # Split into train and val
    num_train = int(N * TRAIN_FRAC)
    num_val = N - num_train
    data_train, data_val = torch.utils.data.random_split(dataset, (num_train, num_val))

    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE)
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE)

    # Sample batch
    x, y = next(iter(dataloader_train))

    model = Transformer(num_classes=C, max_output_length=y.shape[1])
    logits = model(x, y[:, :-1])


    print('x:', x.shape)
    print('y:', y.shape)
    print('logit:', logits.shape)
    print(x[0:1])
    print(model.predict(x[0:1]))
