"""
This file is to select a bag repre from a bag
"""
import torch 
import pdb 
import torch.nn as nn
import torch.nn.functional as F 
import random
from torch.autograd import Variable
from config import Config

class GumbalSoftmax(nn.Module):
    def __init__(self):
        super(GumbalSoftmax, self).__init__()

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, -1)

    def forward(self, logits, temperature):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.att_mat = nn.Parameter(torch.randn(Config.hidden_size * Config.num_feature, Config.rel_num))
        self.rel_mat = nn.Parameter(torch.randn(Config.hidden_size * Config.num_feature, Config.rel_num))
        self.bias = nn.Parameter(torch.randn(Config.rel_num))
        self.softmax = nn.Softmax(-1)
        self.gumbal_softmax = GumbalSoftmax()
        self.dropout = nn.Dropout()


        """init weight"""
        nn.init.xavier_uniform_(self.att_mat)
        nn.init.xavier_uniform_(self.rel_mat)
        nn.init.normal_(self.bias)

        """for mask na relation embedding"""
        random.seed(Config.seed)
        torch.manual_seed(Config.seed)
        self.na_mask = nn.Parameter(torch.ones(Config.rel_num), requires_grad=False)
        self.na_mask[0] = 0
        
    
    def __logit__(self, x):
        return torch.matmul(x, self.rel_mat) + self.bias
    
    def forward(self, x, scope, query = None):
        if Config.training:
            if Config.train_bag:
                if Config.bag_type == "one":
                    bag_repre = []
                    for i in range(scope.shape[0]):
                        bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                        instance_logit = self.softmax(self.__logit__(bag_hidden_mat))
                        j = torch.argmax(instance_logit[:, query[i]])
                        bag_repre.append(bag_hidden_mat[j])
                    bag_repre = torch.stack(bag_repre)
                    bag_repre = self.dropout(bag_repre)
                    bag_logit = self.__logit__(bag_repre)
                    return bag_logit
                elif Config.bag_type == "att":
                    bag_repre = []
                    rel_query = self.rel_mat.transpose(0,1)[query]
                    # attention = self.att_mat.transpose(0,1)[query]
                    att_score = (x * rel_query).sum(-1)
                    for i in range(scope.shape[0]):
                        bag_hidden_mat = x[scope[i][0]:scope[i][1]] # (bag_size, hidden_size)
                        softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]]) #(bag_size)
                        bag_repre.append((softmax_att_score.unsqueeze(-1) * bag_hidden_mat).sum(0))
                    bag_repre = torch.stack(bag_repre, 0)
                    bag_repre = self.dropout(bag_repre)
                    bag_logit = self.__logit__(bag_repre)
                    return bag_logit
            else:
                logit = self.__logit__(x)
                return logit
        else:
            if Config.eval_bag:
                if Config.bag_type == "one":
                    bag_logit = []
                    for i in range(scope.shape[0]):
                        bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                        instance_logit = self.softmax(self.__logit__(bag_hidden_mat))
                        _bag_logit, _ = torch.max(instance_logit, 0)
                        bag_logit.append(_bag_logit)
                    bag_logit = torch.stack(bag_logit)
                    return bag_logit
                elif Config.bag_type == "att":
                    bag_logit = []
                    att_score = torch.matmul(x, self.rel_mat) # (nsum, N)
                    for i in range(scope.shape[0]):
                        bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                        softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1)) # (N, (softmax)n) 
                        rep_for_each_rel = torch.matmul(softmax_att_score, bag_hidden_mat) # (N, n) * (n, H) -> (N, H)
                        logit_for_each_rel = self.softmax(self.__logit__(rep_for_each_rel)) # ((each rel)N, (logit)N)
                        logit_for_each_rel = logit_for_each_rel.diag() # (N)
                        bag_logit.append(logit_for_each_rel)
                    bag_logit = torch.stack(bag_logit, 0)
                    return bag_logit
            else:
                return self.softmax(self.__logit__(x))
                
