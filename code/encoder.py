"""
This file contains many useful neural networks.
"""
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from config import Config
from transformers import BertConfig, BertModel, BertTokenizer

class MLP(nn.Module):
    '''
    MLP network
    The input is [batch_size, *, input_size],
    The output is [batch_size, *, hidden_size]
    '''
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.modules = [
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, output_size),
            nn.ReLU(),
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
        self.net = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(Config.dropout)
        self._minus = -100

    def maxPooling(self, x):
        '''
        x.size(): [batch_size, hidden_size, sen_len] --> [batch_size, hidden_size]
        '''
        text, _ = torch.max(x, -1)
        return text
    
    
    def pieceWiseMaxPooling(self, x, mask):
        '''
        x.size(): `(B, H, L)`
        mask.size: `(B, 3, L)`
        '''
        pool1 = self.maxPooling(self.relu(x + self._minus * mask[:, 0:1, :])) # (B, H)
        pool2 = self.maxPooling(self.relu(x + self._minus * mask[:, 1:2, :]))
        pool3 = self.maxPooling(self.relu(x + self._minus * mask[:, 2:3, :]))
        text = torch.cat([pool1, pool2, pool3], 1) # (B, 3H)
        return text 
    
    def forward(self, input, mask=None):
        if mask is None:
            return self.maxPooling(self.relu(self.net(input)))
        else:
            return self.pieceWiseMaxPooling(self.relu(self.net(input)), mask)

class RNN(nn.Module):
    '''
    RNN network.
    The input is [sen_len, batch, input_size]
    The output of this network is [sen_len, batch_size, num_directions*hidden_size]
    The h_n is [num_layers*num_directions, batch_size, hidden_size]
    '''
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5, bidirectional=False):
        super(RNN, self).__init__()
        self.net = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, input):
        output, (h_n, c_n) = self.net(input)
        return output, h_n

class Bert(nn.Module):
    '''
    Bert model
    '''
    def __init__(self, ):
        super(Bert, self).__init__()
        # pre-train model dict
        self.MODEL_CLASSES = {
            'bert': (BertConfig, BertModel, BertTokenizer),
        }

        # load pretrained model
        config_class, model_class, tokenizer_class = self.MODEL_CLASSES[Config.model_type]
        config = config_class.from_pretrained(Config.model_name_or_path)
        self.tokenizer = tokenizer_class.from_pretrained(Config.model_name_or_path, do_lower_case=True)
        self.model = model_class.from_pretrained(Config.model_name_or_path, from_tf=bool('.ckpt' in Config.model_name_or_path), config=config)

    def forward(self, input_ids, attention_mask, latent=None, mask=None):
        '''
        text: `(batch_size, hidden_size)`
        '''
        inputs = {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'latent':latent,
            'mask':mask
        }

        outputs = self.model(**inputs)
        text = outputs[0][:, 0, :]

        return text

