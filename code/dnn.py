"""
This file contains many useful neural networks.
"""
import torch
import torch.nn as nn
import pdb
from config import Config

class MLP(nn.Module):
    '''
    MLP network
    The input is [batch_size, *, input_size],
    The output is [batch_size, *, hidden_size]
    '''
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.modules = [
            nn.Linear(input_size, output_size),
            # nn.ReLU(),
        ]
        self.net = nn.Sequential(*self.modules)
    
    def forward(self, input):
        return self.net(input)

class CNN(nn.Module):
    '''
    CNN network
    The input is [batch_size, input_size, sen_len],
    The output is [batch_size, hidden_size]
    '''
    def __init__(self, input_size, hidden_size):
        super(CNN, self).__init__()
        self.modules = [
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        self.dropout = nn.Dropout(Config.dropout)
        self.net = nn.Sequential(*self.modules)

    def maxPooling(self, x):
        '''
        x.size(): [batch_size, hidden_size, sen_len] --> [batch_size, hidden_size]
        '''
        text, _ = torch.max(x, -1)
        return text
    
    def forward(self, input):
        return self.dropout(self.maxPooling(self.net(input)))

class RNN(nn.Module):
    '''
    RNN network.
    The input is [sen_len, batch, input_size]
    The output of this network is [sen_len, batch_size, num_directions*hidden_size]
    The h_n is [num_layers*num_directions, batch_size, hidden_size]
    '''
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5, bidirectional=False):
        super(RNN, self).__init__()
        self.modules = [
            nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional),
        ]
        self.net = nn.Sequential(*self.modules)
    
    def forward(self, input):
        output, (h_n, c_n) = self.net(input)
        return output, h_n


