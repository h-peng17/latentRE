"""
This file is to represent the ori text into low dense tensors using dnn
"""
import torch 
import torch.nn as nn 
import numpy as np 
from config import Config
from dnn import *

class TextRepre(nn.Module):
    def __init__(self, word_vec):
        super(TextRepre, self).__init__()
        self.word_embedding = nn.Embedding(word_vec.shape[0], word_vec.shape[1])
        self.word_embedding.weight.data.copy_(torch.from_numpy(word_vec))
        self.word_embedding.weight.data[0].fill_(0)
        self.pos1_embedding = nn.Embedding(Config.pos_num, Config.pos_embedding_dim)
        self.pos2_embedding = nn.Embedding(Config.pos_num, Config.pos_embedding_dim)
        nn.init.xavier_normal_(self.pos1_embedding.weight.data)
        nn.init.xavier_normal_(self.pos2_embedding.weight.data)
        self.cnn = CNN(Config.word_embeeding_dim+Config.pos_embedding_dim*2, Config.hidden_size)
        self.dropout = nn.Dropout(Config.dropout)

    def embedding(self, word, pos1, pos2):
        return torch.cat((self.word_embedding(word), 
                            self.pos1_embedding(pos1),
                                self.pos2_embedding(pos2)), dim=2)
    
    def forward(self, word, pos1, pos2, mask=None):
        embedding = self.embedding(word, pos1, pos2)
        embedding = self.dropout(embedding)
        text = self.cnn(embedding.permute(0,2,1))
        if mask is not None:
            mask_embedding = embedding * (mask.unsqueeze(2))
            entity_info = self.cnn(mask_embedding.permute(0,2,1))
            return text, entity_info
        else:
            return text