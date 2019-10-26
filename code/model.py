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


class LatentRE(nn.Module):
    def __init__(self, word_vec=None, weight=None):
        super(LatentRE, self).__init__()
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
                    labels=None,
                    knowledge=None, 
                    scope=None,
                    query=None):
        if Config.training:
            text = self.encoder(input_ids, attention_mask)
            logit, latent = self.selector(text, scope, query, knowledge)
            ce_loss = self.loss.ce_loss(logit, query)
            kl_loss = self.loss.kl_loss(logit, knowledge)
            if Config.latent:
                gen_loss = self.decoder(input_ids, attention_mask, labels, latent)
                return kl_loss + gen_loss * Config.gen_loss_scale + ce_loss * Config.ce_loss_scale
            else:
                return kl_loss * Config.kl_loss_scale + ce_loss * Config.ce_loss_scale
        else:
            text = self.encoder(input_ids, attention_mask)
            logit = self.selector(text, scope)
            return logit
        # if Config.latent:
        #     latent = self.selector.rel_mat[:, query].transpose(0, 1)
        #     gen_loss = self.decoder(input_ids, attention_mask, token_mask, governor_mask, latent)
        # else:
        #     gen_loss = self.decoder(input_ids, attention_mask, token_mask, governor_mask, None)
        # return gen_loss

