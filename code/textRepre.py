"""
This file is to represent the ori text into low dense tensors using dnn
"""
import torch 
import torch.nn as nn 
import numpy as np 
import pdb
from config import Config
from encoder import *

class TextRepre(nn.Module):
    def __init__(self, word_vec):
        super(TextRepre, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(embeddings=torch.from_numpy(word_vec).to(torch.float32), freeze=False)
        self.pos1_embedding = nn.Embedding(Config.pos_num, Config.pos_embedding_dim)
        self.pos2_embedding = nn.Embedding(Config.pos_num, Config.pos_embedding_dim)
        nn.init.xavier_uniform_(self.pos1_embedding.weight.data)
        nn.init.xavier_uniform_(self.pos2_embedding.weight.data)
        self.pos1_embedding.weight.data[0].fill_(0)
        self.pos2_embedding.weight.data[0].fill_(0)
        self.mask_embedding = nn.Embedding(4,3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.mask_embedding.weight.requires_grad = False
        self.cnn = CNN(Config.word_embeeding_dim+Config.pos_embedding_dim*2, Config.hidden_size)


    def embedding(self, word, pos1, pos2):
        return torch.cat((self.word_embedding(word), 
                            self.pos1_embedding(pos1),
                                self.pos2_embedding(pos2)), dim=2)
    
    def forward(self, word, pos1, pos2, mask=None):
        embedding = self.embedding(word, pos1, pos2)
        if mask is None:
            text = self.cnn(embedding.permute(0,2,1))
        else:
            mask = 1 - self.mask_embedding(mask).transpose(1, 2) # (B, L) -> (B, L, 3) -> (B, 3, L)
            text = self.cnn(embedding.permute(0,2,1), mask)
        return text
        