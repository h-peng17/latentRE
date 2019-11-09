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
        checkpoint = torch.load(os.path.join(Config.save_path, "ckptencoder_not_na9"))
        self.encoder = Bert()
        self.encoder.load_state_dict(checkpoint["encoder"])
        for param in self.encoder.parameters():
            param.requires_grad = False # frozen
        self.selector = Selector()
        self.selector.load_state_dict(checkpoint['selector'])
        for param in self.selector.parameters():
            param.requires_grad = False # frozen

        decoder_ckpt = torch.load(os.path.join(Config.save_path, "ckptlatent29"))
        self.decoder = BertDecoder()
        self.decoder.load_state_dict(decoder_ckpt['decoder'])
        self.decoder_rel_mat = nn.Parameter(torch.zeros(Config.hidden_size, Config.rel_num))
        self.decoder_rel_mat.load_state_dict(decoder_ckpt['decoder_rel_mat'])
        self.loss = Loss(weight)
        
    def forward(self, 
                  word=None,
                  pos1=None,
                  pos2=None,
                  label=None,
                  input_ids=None, 
                  attention_mask=None, 
                  mask=None,
                  query=None,
                  knowledge=None, 
                  scope=None):
        if Config.training:
            text = self.encoder(input_ids, attention_mask)
            logit, gumbal_logit = self.selector(text, scope, query)
            latent = torch.matmul(gumbal_logit, self.decoder_rel_mat.transpose(0, 1))
            ce_loss = self.loss.ce_loss(logit, query)
            kl_loss = self.loss.kl_loss(logit, knowledge)
            if Config.latent:
                gen_loss = self.decoder(input_ids, attention_mask, mask, latent)
                return kl_loss * Config.kl_loss_scale + gen_loss * Config.gen_loss_scale + ce_loss * Config.ce_loss_scale
            else:
                return kl_loss * Config.kl_loss_scale + ce_loss * Config.ce_loss_scale, torch.argmax(logit, 1)
        else:
            text = self.encoder(input_ids, attention_mask)
            logit = self.selector(text, scope)
            return logit

            
        


