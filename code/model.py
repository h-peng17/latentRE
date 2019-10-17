"""
model
"""
import torch
import torch.nn as nn 
import pdb 
import torch.nn.functional as F
from torch.autograd import Variable
from config import Config
from textRepre import TextRepre
from selector import Selector
from loss import Loss
from decoder import BertDecoder
from encoder import Bert


class LatentRE(nn.Module):
    def __init__(self, word_vec=None, weight=None):
        super(LatentRE, self).__init__()
        self.textRepre = TextRepre(word_vec)
        self.encoder = Bert()
        self.selector = Selector()
        self.decoder = BertDecoder()
        self.loss = Loss(weight)
    
    def forward(self, 
                    pos_word=None, 
                    pos_pos1=None, 
                    pos_pos2=None, 
                    input_ids=None, 
                    attention_mask=None,  
                    knowledge=None, 
                    scope=None,
                    query=None):
        if Config.training:
            # text = self.textRepre(pos_word, pos_pos1, pos_pos2)
            text = self.encoder(input_ids, attention_mask)
            logit, latent = self.selector(text, scope, query)
            kl_loss = self.loss.kl_loss(logit, knowledge)
            if Config.latent:
                gen_loss = self.decoder(input_ids, attention_mask, latent)
                return kl_loss + gen_loss * Config.loss_scale
            else:
                return kl_loss
        else:
            # text = self.textRepre(pos_word, pos_pos1, pos_pos2)
            text = self.encoder(input_ids, attention_mask)
            logit = self.selector(text, scope)
            return logit

