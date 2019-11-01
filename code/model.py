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
    def __init__(self, word_vec, weight=None):
        super(LatentRE, self).__init__()
        # self.textRepre = TextRepre(word_vec)
        self.encoder = Bert()
        self.selector = Selector()
        self.decoder = BertDecoder()
        self.loss = Loss(weight)
        self.decoder_margin = BertDecoder()
        
    def forward(self, 
                  input_ids=None, 
                  attention_mask=None, 
                  decoder_input_ids=None,
                  decoder_attention_mask=None,
                  mask=None,
                  query=None,
                  knowledge=None, 
                  scope=None):
        if Config.training:
            text = self.encoder(input_ids, attention_mask)
            logit, latent = self.selector(text, scope, query)
            ce_loss = self.loss.ce_loss(logit, query)
            kl_loss = self.loss.kl_loss(logit, knowledge)
            if Config.latent:
                gen_loss = self.decoder(decoder_input_ids, decoder_attention_mask, mask, latent)
                margin_gen_loss = self.decoder_margin(decoder_input_ids, decoder_attention_mask, mask, None)
                return max(0, kl_loss + gen_loss * Config.gen_loss_scale + ce_loss * Config.ce_loss_scale - margin_gen_loss + 2.0)
            else:
                return kl_loss * Config.kl_loss_scale + ce_loss * Config.ce_loss_scale
        else:
            text = self.encoder(input_ids, attention_mask)
            logit = self.selector(text, scope)
            return logit



