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
import time
import os



class LatentRE(nn.Module):
    def __init__(self, word_vec, weight=None):
        super(LatentRE, self).__init__()
        ''' load encoder '''
        if Config.encoder == "bert":
            self.encoder = Bert()
        else:
            self.encoder = TextRepre(word_vec)
        self.selector = Selector()
        self.loss = Loss(weight)

    def forward(self, 
                  word=None,
                  pos1=None,
                  pos2=None,
                  label=None,
                  pcnn_mask=None,
                  input_ids=None, 
                  attention_mask=None, 
                  query=None,
                  scope=None):
        if Config.training:
            if Config.encoder == "bert":
                text = self.encoder(input_ids, attention_mask)
                logit = self.selector(text, None)
                ce_loss = self.loss.ce_loss(logit, query)
            elif Config.encoder == "pcnn":
                text = self.encoder(word, pos1, pos2, pcnn_mask)
                if Config.bag_type == "one":
                    logit = self.selector(text, scope, label)
                elif Config.bag_type == "att":
                    logit = self.selector(text, scope, query)
                ce_loss = self.loss.ce_loss(logit, label)
            elif Config.encoder == "cnn":
                text = self.encoder(word, pos1, pos2)
                if Config.bag_type == "one":
                    logit = self.selector(text, scope, label)
                elif Config.bag_type == "att":
                    logit = self.selector(text, scope, query)
                ce_loss = self.loss.ce_loss(logit, label)
            return ce_loss
        else:
            if Config.encoder == "bert":
                text = self.encoder(input_ids, attention_mask)
            elif Config.encoder == "pcnn":
                text = self.encoder(word, pos1, pos2, pcnn_mask)
            elif Config.encoder == "cnn":
                text = self.encoder(word, pos1, pos2)
            logit = self.selector(text, scope)
            return logit
            


           
            
        


