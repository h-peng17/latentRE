
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import pdb 
import sklearn.metrics
import matplotlib
import os 
# Use 'Agg' so this program could run on a remote server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import Config



class BagTest(object):
    '''
    # Because bag test is memory-overflow
    # so I add this class 
    '''
    def __init__(self, entpair2scope, query):
        # logit.size(): `(ins_tot, rel_num)`
        # scope.size(): `(num_bag, 2)`
        # label.size(): `()`
        # multi_label.size(): `(num_bag, rel_num)` 
        
        self.logit = torch.zeros((0, Config.rel_num))
        entpair_tot = len(entpair2scope)
        # scope
        self.scope = []
        for key in entpair2scope.keys():
            self.scope.append(entpair2scope[key])
        # label
        _label = []
        for i in range(entpair_tot):
            _label.append(query[self.scope[i][0]])
        self.label = np.stack(_label)
        # multi_label
        _multi_label = []
        for i in range(entpair_tot):
            _one_multi_label = np.zeros((Config.rel_num), dtype=np.int32)
            for j in range(self.scope[i][0], self.scope[i][1]):
                _one_multi_label[query[j]] = 1
            _multi_label.append(_one_multi_label)
        self.multi_label = np.stack(_multi_label)
            
        self.auc = 0
        self.epoch = 0
    
    def update(self, logit):
        self.logit = torch.cat((self.logit, logit),0)
    
    def clean(self):
        self.logit = torch.zeros((0, Config.rel_num))

    def eval(self, logit, label):
        res_list = []
        tot = 0
        for i in range(len(logit)):
            for j in range(1, len(logit[i])):
                tot += label[i][j]
                res_list.append([logit[i][j], label[i][j]])
                
        #sort res_list
        res_list.sort(key=lambda val: val[0], reverse=True)
        precision = np.zeros((len(res_list)), dtype=float)
        recall = np.zeros((len(res_list)), dtype=float)
        corr = 0
        for i, res in enumerate(res_list):
            corr += res[1]
            precision[i] = corr/(i+1)
            recall[i] = corr/tot
        
        # pdb.set_trace()
        f1 = (2*precision*recall / (recall+precision+1e-20)).max()
        auc = sklearn.metrics.auc(x=recall, y=precision)
        print("auc = "+str(auc)+"| "+"F1 = "+str(f1))
        print('P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(precision[100], precision[200], precision[300], (precision[100] + precision[200] + precision[300]) / 3))
        
        if auc > self.auc:
            plt.plot(recall, precision, lw=2, label=Config.info)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.3, 1.0])
            plt.xlim([0.0, 0.4])
            plt.title('Precision-Recall')
            plt.legend(loc="upper right")
            plt.grid(True)
            plt.savefig(os.path.join(Config.save_path, 'pr_curve_' + Config.info + '.png'))

        return auc
    
    def forward(self, epoch):
        # if self.logit.size()[0] != len(self.scope):
        #     exit("--------------------------wrong! The test data is not aligned!------------------------")
        bag_logit = []
        for i in range(len(self.scope)):
            bag = self.logit[self.scope[i][0]:self.scope[i][1]]
            _bag_logit, _ = torch.max(bag, 0)
            bag_logit.append(_bag_logit)
        
        # `(num_bag, rel_num)`
        bag_logit = torch.stack(bag_logit)
        output = torch.argmax(bag_logit, 1).numpy()
        bag_logit = bag_logit.numpy().tolist()

        # compute acc
        tot = self.label.shape[0]
        tot_na = (self.label==0).sum()
        tot_not_na = tot - tot_na
        tot_corr = (self.label==output).sum()
        na_corr = np.logical_and(self.label==output, self.label==0).sum()
        not_na_corr = tot_corr - na_corr
        print("dev: acc:%.3f, na_acc:%.3f, not_na_acc:%.3f" % (tot_corr/tot, na_corr/tot_na, not_na_corr/tot_not_na))

        # compute auc
        auc = self.eval(bag_logit, self.multi_label)
        if auc > self.auc:
            self.auc = auc
            self.epoch = epoch
        
        self.clean()






        

