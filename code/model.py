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



class LatentRE(nn.Module):
    def __init__(self, word_vec, weight):
        super(LatentRE, self).__init__()
        self.textRepre = TextRepre(word_vec)
        self.selector = Selector()
        self.decoder = BertDecoder()
        self.loss = Loss(weight)
    
    def forward(self, 
                    pos_word, 
                    pos_pos1, 
                    pos_pos2, 
                    input_ids=None, 
                    attention_mask=None,  
                    knowledge=None, 
                    scope=None,
                    query=None):
        if Config.training:
            text = self.textRepre(pos_word, pos_pos1, pos_pos2)
            logit, label_info = self.selector(text, scope, query)
            kl_loss = self.loss.kl_loss(logit, knowledge)
            # util bert decoder with lantent text and addtional info `label_info` as 'type'
            gen_loss = self.decoder(input_ids, text, label_info, attention_mask)
            return kl_loss+gen_loss
        else:
            text = self.textRepre(pos_word, pos_pos1, pos_pos2)
            logit = self.selector(text, scope)
            return logit, torch.argmax(logit, 1)

