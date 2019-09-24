"""
This file is to encode-decode text and calculate loss
"""
import torch 
import torch.nn as nn
from dnn import *
from config import Config
import pdb
 
class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, rel_num):
        super(EncoderDecoder, self).__init__()
        self.encoder = MLP(hidden_size, rel_num, hidden_size)
        self.decoder = MLP(rel_num+hidden_size, hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, text, entity_info=None):
        text_latent = self.encoder(text)
        if entity_info is None:
            return self.softmax(text_latent)
        else:
            latent = torch.cat((entity_info, text_latent), dim=1)
            generated_text = self.decoder(latent)
            return text_latent, generated_text


class Loss(nn.Module):
    def __init__(self, weight):
        super(Loss, self).__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean")   
        self.crossEntropy = nn.CrossEntropyLoss(weight=torch.from_numpy(weight).to(torch.float32))
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, text, neg_samples, generated_text, text_latent, knowledge, label):
        # ce-loss:
        # ce_loss = self.crossEntropy(text_latent, label)

        # # kl-loss:
        text_latent = self.softmax(text_latent)
        kl_loss = self.kl(text_latent.clamp(min=1e-10).log(), knowledge).sum(0)

        # # generate-loss:
        # generated_text = self.softmax(generated_text)
        # text = self.softmax(text)
        # neg_samples = self.softmax(neg_samples)
        # pos_generate_loss = self.kl(generated_text.clamp(min=1e-10).log(), text).sum(0)
        # neg_samples = neg_samples.view(text.size()[0], Config.neg_samples, Config.hidden_size)
        # generated_text = generated_text.unsqueeze(1).expand([text.size()[0], Config.neg_samples, Config.hidden_size])
        # neg_generate_loss = self.kl(generated_text.clamp(min=1e-10).log(), neg_samples).sum(0)

        # return kl_loss + pos_generate_loss - (1/Config.neg_samples) * neg_generate_loss
        # return kl_loss + pos_generate_loss
        # return ce_loss
        return kl_loss
        
        