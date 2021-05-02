#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        #print("chomp input = " + str(x.size()))
        result = x[:, :, :-self.chomp_size].contiguous()
        #print("chomp output = " + str(result.size()))
        return result


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        #self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size,
        #                                   stride=stride, padding=padding, dilation=dilation))
        
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #print("block input size = " + str(x.size()))
        #print("Weights for conv1 = " + str(self.conv1.weight.size()))

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        result = self.relu(out + res)
        #print("temp block output size = " + str(result.size()))
        return result


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        #print("temporal conv net input = " + str(x.size()))
        result = self.network(x)
        #print("temporal conv net output size = " + str(result.size()))
        return result

class TCN(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_channels, num_channels, kernel_size=kernel_size, dropout=dropout)
#         self.linear = nn.Linear(num_qs * num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        #print("tcn input size = " + str(inputs.size()))
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
#         o = self.linear(y1[:, :, -1].view(1, -1))
        #print("before cutting = " + str(y1.size()))
        o = y1[:, :, -1]
        #print("tcn output size = " + str(o.size()))
        return o
#         return F.log_softmax(o, dim=1)