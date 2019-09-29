"""
# This file is to calculate loss 
"""
import torch
import torch.nn as nn
from config import Config

class Loss(nn.Module):
    def __init__(self, weight):
        super(Loss, self).__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean")   
        self.crossEntropy = nn.CrossEntropyLoss(weight=torch.from_numpy(weight).to(torch.float32))
        self.softmax = nn.Softmax(dim=1)
    
    def ce_loss(self, logit, label):
        ce_loss = self.crossEntropy(logit, label)
        return ce_loss

    def rel_pre_loss(self, logit, knowledge):
        # kl-loss:
        kl_loss = self.kl(logit.clamp(min=1e-10).log(), knowledge).sum(0)
        return kl_loss
    
    def gen_loss(self, golden_text, neg_samples, generated_text):
        # cal distribution
        generated_text = self.softmax(generated_text)
        golden_text = self.softmax(golden_text)
        neg_samples = self.softmax(neg_samples)
        # pos loss
        pos_gen_loss = self.kl(generated_text.clamp(min=1e-10).log(), golden_text).sum(0)
        # neg loss
        neg_samples = neg_samples.view(golden_text.size()[0], Config.neg_samples, Config.hidden_size)
        generated_text = generated_text.unsqueeze(1).expand([golden_text.size()[0], Config.neg_samples, Config.hidden_size])
        neg_gen_loss = self.kl(generated_text.clamp(min=1e-10).log(), neg_samples).sum(0)
        return pos_gen_loss
    
    def forward(self):
        pass 
        