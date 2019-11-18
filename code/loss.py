"""
# This file is to calculate loss 
"""
import pdb 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from config import Config

class Loss(nn.Module):
    def __init__(self, weight):
        super(Loss, self).__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean")   
        # weight=torch.from_numpy(weight).to(torch.float32)
        self.crossEntropy = nn.CrossEntropyLoss(weight=torch.from_numpy(weight).to(torch.float32), reduction="none")
        self.neg_crossEntropy = nn.CrossEntropyLoss(reduction='sum')
        self.bceloss = nn.BCEWithLogitsLoss()
        self.softmax = nn.Softmax(dim=1)

    
    def ce_loss(self, logit, label, bce_weight):
        ce_loss = self.crossEntropy(logit, label)
        ce_loss = (ce_loss * bce_weight).mean()
        return ce_loss
    
    def ce_loss_neg(self, logit, label):
        ce_loss = self.neg_crossEntropy(logit, label)
        return ce_loss
    
    def bce_loss(self, logit, label):
        bce_loss = self.bceloss(logit, label)
        return bce_loss

    def kl_loss(self, logit, knowledge):
        # kl-loss:
        kl_loss = self.kl(F.log_softmax(logit, dim=-1), knowledge).sum(0)
        return kl_loss

    

        
        