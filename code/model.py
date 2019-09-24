"""
model
"""
import torch
import torch.nn as nn 
import pdb 
# from torch.utils.tensorboard import SummaryWriter
from config import Config
from textRepre import TextRepre
from latent import Loss, EncoderDecoder

class LatentRE(nn.Module):
    def __init__(self, word_vec, weight):
        super(LatentRE, self).__init__()
        self.textRepre = TextRepre(word_vec)
        self.encoderDecoder = EncoderDecoder(Config.hidden_size, Config.rel_num)
        self.loss = Loss(weight)
        self.pos_word = None
        self.pos_pos1 = None
        self.pos_pos2 = None 
        self.neg_word = None
        self.neg_pos1 = None 
        self.neg_pos2 = None
        self.mask = None 
        self.knowledge = None
        self.label = None
    
    def forward(self):
        text, entity_info = self.textRepre(self.pos_word, self.pos_pos1, self.pos_pos2, self.mask)
        neg_samples = self.textRepre(self.neg_word.view(-1, Config.sen_len), 
                                        self.neg_pos1.view(-1, Config.sen_len),
                                            self.neg_pos2.view(-1, Config.sen_len))
        text_latent, generated_text = self.encoderDecoder(text, entity_info)
        loss = self.loss(text, neg_samples, generated_text, text_latent, self.knowledge, self.label)
        return loss, torch.argmax(text_latent, 1)
    
    def test(self):
        text = self.textRepre(self.pos_word, self.pos_pos1, self.pos_pos2)
        text_latent = self.encoderDecoder(text)
        return text_latent, torch.argmax(text_latent, 1)