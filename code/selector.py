"""
This file is to select a bag repre from a bag
"""
import torch 
import torch.nn as nn
from config import Config

class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.rel_mat = nn.Parameter(torch.randn(Config.hidden_size, Config.rel_num))
        self.bias = nn.Parameter(torch.randn(Config.rel_num))
        self.softmax = nn.Softmax(1)
    
    def __logit__(self, x):
        return torch.matmul(x, self.rel_mat) + self.bias
    
    def forward(self, x, scope, query = None, knowledge=None):
        if Config.training:
            if Config.train_bag:
                bag_repre = []
                for i in range(scope.shape[0]):
                    bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                    instance_logit = self.softmax(self.__logit__(bag_hidden_mat))
                    j = torch.argmax(instance_logit[:, query[i]])
                    bag_repre.append(bag_hidden_mat[j])
                bag_repre = torch.stack(bag_repre)
                return self.__logit__(bag_repre), self.rel_mat
            else:
                return self.softmax(self.__logit__(x)), self.rel_mat     
        else:
            if Config.eval_bag:
                bag_logit = []
                for i in range(scope.shape[0]):
                    bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                    instance_logit = self.softmax(self.__logit__(bag_hidden_mat))
                    _bag_logit, _ = torch.max(instance_logit, 0)
                    bag_logit.append(_bag_logit)
                bag_logit = torch.stack(bag_logit)
                return bag_logit
            else:
                return self.softmax(self.__logit__(x))
                
