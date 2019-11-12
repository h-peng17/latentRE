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
        # checkpoint = torch.load(os.path.join(Config.save_path, "ckptencoder1"))
        # self.encoder = Bert()
        # self.encoder.load_state_dict(checkpoint["encoder"])
        # for param in self.encoder.parameters():
            # param.requires_grad = False # frozen
        # self.selector = Selector()
        # for param in self.selector.parameters():
        #     param.requires_grad = False # frozen

        # decoder_ckpt = torch.load(os.path.join(Config.save_path, "ckptlatent39"))
        # self.selector.load_state_dict(decoder_ckpt['selector'])
        # self.decoder = BertDecoder()
        # self.decoder.load_state_dict(decoder_ckpt['decoder'])
        # for param in self.decoder.parameters():
            # param.requires_grad = False
        # self.selector.decoder_rel_mat.requires_grad = False
        # self.loss = Loss(weight)
        self.encoder = TextRepre(word_vec)
        self.selector = Selector()
        self.loss = Loss(weight)

    def forward(self, 
                  pos_word=None,
                  pos_pos1=None,
                  pos_pos2=None,
                  pos_label=None,
                  pos_query=None,
                  pos_scope=None,
                  neg_word=None,
                  neg_pos1=None,
                  neg_pos2=None,
                  neg_label=None,
                  neg_scope=None,
                  mul_label=None,  
                  mul_num=None, 
                  label=None,
                  input_ids=None, 
                  attention_mask=None, 
                  mask=None,
                  query=None,
                  knowledge=None, 
                  scope=None):
        # if Config.training:
        #     text = self.encoder(input_ids, attention_mask)
        #     logit, latent = self.selector(text, scope, query)
        #     ce_loss = self.loss.ce_loss(logit, query)
        #     kl_loss = self.loss.kl_loss(logit, knowledge)
        #     if Config.latent:
        #         gen_loss = self.decoder(input_ids, attention_mask, mask, latent)
        #         return kl_loss * Config.kl_loss_scale + gen_loss * Config.gen_loss_scale + ce_loss * Config.ce_loss_scale
        #     else:
        #         return kl_loss * Config.kl_loss_scale + ce_loss * Config.ce_loss_scale, torch.argmax(logit, 1)
        # else:
        #     text = self.encoder(input_ids, attention_mask)
        #     logit = self.selector(text, scope)
        #     return logit
        if Config.training:
            '''logit shape `(batch_size, rel_num)` 
               pos_label shape `(batch_size, rel_num)`'''
            Config.train_bag = True
            pos_text = self.encoder(pos_word, pos_pos1, pos_pos2)
            pos_logit = self.selector(pos_text, pos_scope, pos_query)
            pos_logit = F.softmax(pos_logit, 1)
            pos_score = torch.mean(torch.sum(pos_logit * pos_label.float(), 1)) #should be 1

            Config.train_bag = False
            neg_text = self.encoder(neg_word, neg_pos1, neg_pos2)
            neg_logit, _ = self.selector(neg_text, None, None)
            neg_logit = F.softmax(neg_logit, 1)
            neg_score = torch.mean(torch.sum(neg_logit * mul_label.float(), 1)/mul_num.float()) # should be 0
            neg_pos_score = torch.mean(torch.sum(neg_logit * neg_label.float(), 1)) # should be 1


            return neg_score - pos_score - neg_pos_score + 2.0
        else:
            text = self.encoder(pos_word, pos_pos1, pos_pos2)
            logit = self.selector(text, None, None)
            return logit

           
            
        


