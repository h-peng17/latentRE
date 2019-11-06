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
from decoder import BertDecoder, GPT2Decoder
from encoder import Bert
import time


class LatentRE(nn.Module):
    def __init__(self, word_vec, weight=None):
        super(LatentRE, self).__init__()
        self.encoder = Bert()
        self.selector = Selector()
        # self.decoder = BertDecoder()
        self.decoder = GPT2Decoder()
        self.loss = Loss(weight)
        
    def forward(self, 
                  word=None,
                  pos1=None,
                  pos2=None,
                  label=None,
                  input_ids=None, 
                  attention_mask=None, 
                  decoder_input_ids=None,
                  decoder_attention_mask=None,
                  mask=None,
                  labels=None,
                  query=None,
                  knowledge=None, 
                  scope=None):
        if Config.training:
            text = self.encoder(input_ids, attention_mask)
            logit, latent = self.selector(text, scope, query)
            ce_loss = self.loss.ce_loss(logit, query)
            kl_loss = self.loss.kl_loss(logit, knowledge)
            if Config.latent:
                gen_loss, pre_words = self.decoder(decoder_input_ids, decoder_attention_mask, mask, latent, labels)
                return kl_loss + gen_loss * Config.gen_loss_scale, pre_words # !!!!!!
            else:
                return kl_loss * Config.kl_loss_scale + ce_loss * Config.ce_loss_scale
        else:
            text = self.encoder(input_ids, attention_mask)
            logit = self.selector(text, scope)
            return logit

        


