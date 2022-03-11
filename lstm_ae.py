
#%%
from typing import DefaultDict
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from models import ConvLSTMCell
from torch.autograd import Function

class EncoderDecoderConvLSTM(nn.Module):

    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=3,
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=3,
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=3,
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=3,
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

        self.layer_norm = nn.LayerNorm([64, 50, 50])

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here

            # h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
            #                                      cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        # encoder_vector = h_t2
        encoder_vector = h_t
        b, c, h, w = encoder_vector.shape
        out_encoder_vector = encoder_vector.view(b, -1)

        # decoder
        for t in range(future_step):

            h_t2, c_t2 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

            # h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
            #                                      cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here

            encoder_vector = h_t2

            outputs += [h_t2]  # predictions

            # h_t3, c_t3 = self.decoder_2_convlstm(input_tensor=h_t2,
            #                                      cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = outputs.permute(0, 2, 1, 3, 4)

        return outputs, out_encoder_vector

    def forward(self, x, future_step):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states

        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs, encorded_vector = self.autoencoder(x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs, encorded_vector

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
