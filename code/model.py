"""
model
"""
import torch
import torch.nn as nn 
import pdb 
# from torch.utils.tensorboard import SummaryWriter
from config import Config
from textRepre import TextRepre
from selector import Selector
from decoder import Decoder
from loss import Loss

class LatentRE(nn.Module):
    def __init__(self, word_vec, weight):
        super(LatentRE, self).__init__()
        self.textRepre = TextRepre(word_vec)
        self.selector = Selector()
        self.decoder = Decoder(Config.hidden_size, Config.rel_num)
        self.loss = Loss(weight)
        self.pos_word = None
        self.pos_pos1 = None
        self.pos_pos2 = None 
        self.neg_word = None
        self.neg_pos1 = None 
        self.neg_pos2 = None
        self.mask = None 
        self.knowledge = None
        self.bag_knowledge = None
        self.scope = None # numpy
    
    def forward(self):
        text, ent_context = self.textRepre(self.pos_word, self.pos_pos1, self.pos_pos2, self.mask)
        neg_samples = self.textRepre(self.neg_word.view(-1, Config.sen_len), 
                                        self.neg_pos1.view(-1, Config.sen_len),
                                            self.neg_pos2.view(-1, Config.sen_len))
        logit = self.selector(text, self.scope, self.knowledge)
        # generated_text = self.decoder(logit, ent_context)
        rel_pre_loss = self.loss.rel_pre_loss(logit, self.bag_knowledge)
        # gen_loss = self.loss.gen_loss(text, neg_samples, generated_text)
        return rel_pre_loss, torch.argmax(logit, 1)
    
    def test(self):
        text = self.textRepre(self.pos_word, self.pos_pos1, self.pos_pos2)
        logit = self.selector(text, self.scope)
        return logit, torch.argmax(logit, 1)