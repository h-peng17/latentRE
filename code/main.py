"""
Train
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import argparse
import sklearn.metrics
import matplotlib
import pdb
import numpy as np 
# Use 'Agg' so this program could run on a remote server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import LatentRE
from config import Config
from dataloader import Dataloader
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def eval(logit, label):
    res_list = []
    for i in range(len(label)):
        for j in range(1, len(logit[i])):
            flag = 0
            if j == label[i]:
                flag = 1
            res_list.append([logit[i][j], flag])
    #sort res_list
    res_list.sort(key=lambda val: val[0], reverse=True)

    tot = len(res_list)
    precision = np.zeros((len(res_list)), dtype=float)
    recall = np.zeros((len(res_list)), dtype=float)
    corr = 0
    for i, res in enumerate(res_list):
        corr += res[1]
        precision[i] = corr/tot
        recall[i] = corr/(i+1)
    
    f1 = (2*precision*recall / (x+y+1e-20)).max()
    auc = sklearn.metrics.auc(x=recall, y=precision)
    print("auc = "+str(auc)+"|\t"+"F1 = "+str(f1))
    print('P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(precision[100], precision[200], precision[300], (precision[100] + precision[200] + precision[300]) / 3))
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.4])
    plt.title('Precision-Recall')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, 'pr_curve'))

    


def to_tensor(array):
    return torch.from_numpy(array).to(torch.int64).cuda()

def train(model, train_dataloader, dev_dataloader=None):
    model.cuda()
    model.train()
    params = filter(lambda x:x.requires_grad, model.parameters())
    optimizer = optim.SGD(params, Config.lr)
    tot = 0
    tot_correct = 0
    not_na_correct = 0
    print("Begin train...")
    for i in range(Config.max_epoch):
        # set train data
        # pdb.set_trace()
        for j in range(int(train_dataloader.instance_tot / Config.batch_size)):
            batch_data = train_dataloader.next_batch()
            model.pos_word = to_tensor(batch_data["pos_word"])
            model.pos_pos1 = to_tensor(batch_data["pos_pos1"])
            model.pos_pos2 = to_tensor(batch_data["pos_pos2"])
            model.neg_word = to_tensor(batch_data["neg_word"])
            model.neg_pos1 = to_tensor(batch_data["neg_pos1"])
            model.neg_pos2 = to_tensor(batch_data["neg_pos2"])
            model.label = to_tensor(batch_data["label"])
            model.mask = torch.from_numpy(batch_data["mask"]).to(torch.float32).cuda()
            model.knowledge = torch.from_numpy(batch_data["knowledge"]).to(torch.float32).cuda()
            label = batch_data["label"]
            # train 
            optimizer.zero_grad()
            loss, output = model()
            loss.backward()
            # gen res
            output = output.cpu().detach().numpy()
            tot += label.shape[0]
            tot_correct += (label==output).sum()
            not_na_correct += np.logical_and(label==output, label!=0).sum()
            sys.stdout.write("epoch:%d, loss:%.3f, acc:%.3f, not_na_acc:%.3f\r\n"%(i, loss, tot_correct/tot, not_na_correct/tot))
            sys.stdout.flush()
        
        if i % Config.dev_step == 0:
            pass 




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--cuda", dest="cuda", type=str, default="7", help="cuda")
    args = parser.parse_args()
    
    # set para
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    
    # train
    train_dataloader = Dataloader("train")
    dev_dataloader = Dataloader("test")
    model = LatentRE(train_dataloader.word_vec, train_dataloader.weight)
    train(model, train_dataloader, dev_dataloader)

        

        


        
        

        
        
    