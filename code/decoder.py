"""
This file is to encode-decode text and calculate loss
"""
import torch 
import torch.nn as nn
import pdb
from dnn import *
from config import Config
from selector import Selector

class Decoder(nn.Module):
    def __init__(self, hidden_size, rel_num):
        super(Decoder, self).__init__()
        self.decoder = MLP(2 * hidden_size, hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, select_mask, rel_mat, ent_context):
        latent = torch.cat((torch.matmul(select_mask, rel_mat.transpose(0,1)), ent_context), 1)
        generated_text = self.decoder(latent)
        return generated_text



        
        
