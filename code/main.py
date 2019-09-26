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
    print(Config.loss_func + "  auc = "+str(auc)+"| "+"F1 = "+str(f1))
    print('P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(precision[100], precision[200], precision[300], (precision[100] + precision[200] + precision[300]) / 3))
    
    plt.plot(recall, precision, lw=2, label="model")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.4])
    plt.title('Precision-Recall')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(Config.save_path, 'pr_curve_' + Config.loss_func))

    


def to_tensor(array):
    return torch.from_numpy(array).to(torch.int64).cuda()

def train(model, train_dataloader, dev_dataloader=None):
    model.cuda()
    model.train()
    params = filter(lambda x:x.requires_grad, model.parameters())
    optimizer = optim.SGD(params, Config.lr)
    print("Begin train...")
    for i in range(Config.max_epoch):
        # set train data
        # pdb.set_trace()
        Config.training = True
        tot = 0
        tot_na = 0
        tot_not_na = 0
        tot_correct = 0
        na_correct = 0
        not_na_correct = 0
        for j in range(int(train_dataloader.entpair_tot / Config.batch_size)):
            batch_data = train_dataloader.next_batch()
            model.pos_word = to_tensor(batch_data["pos_word"])
            model.pos_pos1 = to_tensor(batch_data["pos_pos1"])
            model.pos_pos2 = to_tensor(batch_data["pos_pos2"])
            model.neg_word = to_tensor(batch_data["neg_word"])
            model.neg_pos1 = to_tensor(batch_data["neg_pos1"])
            model.neg_pos2 = to_tensor(batch_data["neg_pos2"])
            model.mask = torch.from_numpy(batch_data["mask"]).to(torch.float32).cuda()
            model.knowledge = torch.from_numpy(batch_data["knowledge"]).to(torch.float32).cuda()
            model.bag_knowledge = torch.from_numpy(batch_data["bag_knowledge"]).to(torch.float32).cuda()
            model.scope = batch_data["scope"]
            label = batch_data["label"]
            # train 
            optimizer.zero_grad()
            loss, output = model()
            loss.backward()
            optimizer.step()
            # gen res
            output = output.cpu().detach().numpy()
            tot += label.shape[0]
            tot_na += (label==0).sum()
            tot_not_na += (label!=0).sum()
            tot_correct += (label==output).sum()
            na_correct +=np.logical_and(label==output, label==0).sum()
            not_na_correct += np.logical_and(label==output, label!=0).sum()
            sys.stdout.write("train:epoch:%d, loss:%.3f, acc:%.3f, na_acc:%.3f, not_na_acc:%.3f\r"%(i, loss, tot_correct/tot, na_correct/tot_na, not_na_correct/tot_not_na))
            sys.stdout.flush()
        print("")
        
        if i % Config.dev_step == 0:
            print("begin deving...")
            Config.training = False
            model.eval()
            logits = []
            labels = []
            tot = 0
            tot_na = 0
            tot_not_na = 0
            tot_correct = 0
            na_correct = 0
            not_na_correct = 0
            for j in range(int(dev_dataloader.entpair_tot / Config.batch_size)):
                batch_data = dev_dataloader.next_batch()
                model.pos_word = to_tensor(batch_data["pos_word"])
                model.pos_pos1 = to_tensor(batch_data["pos_pos1"])
                model.pos_pos2 = to_tensor(batch_data["pos_pos2"])
                model.scope = batch_data["scope"]
                label = batch_data["label"]
                multi_label = batch_data["multi_label"]
                logit, output = model.test()
                logits.extend(logit.cpu().detach().numpy().tolist())
                labels.extend(multi_label.tolist())
                output = output.cpu().detach().numpy()
                tot += label.shape[0]
                tot_na += (label==0).sum()
                tot_not_na += (label!=0).sum()
                tot_correct += (label==output).sum()
                na_correct +=np.logical_and(label==output, label==0).sum()
                not_na_correct += np.logical_and(label==output, label!=0).sum()
                sys.stdout.write("dev:epoch:%d, acc:%.3f, na_acc:%.3f, not_na_acc:%.3f\r"%(i, tot_correct/tot, na_correct/tot_na, not_na_correct/tot_not_na))
                sys.stdout.flush()
            print("")
            eval(logits, labels)
            print("---------------------------------------------------------------------------------------------------")
            model.train()
        
        # if i % Config.save_epoch == 0:
            # torch.save(model.state_dict(), os.path.join(Config.save_path, "ckpt"+str(i)))


def test(model, test_dataloader):
    model.cuda()
    print("begin testing...")
    Config.training = False
    for i in range(Config.max_epoch):
        if not os.path.exists(os.path.join(Config.save_path, "ckpt"+str(i))):
            continue
        model.load_static_dict(torch.load(os.path.join(Config.save_path, "ckpt"+str(i))))
        model.eval()
        logits = []
        labels = []
        tot = 0
        tot_na = 0
        tot_not_na = 0
        tot_correct = 0
        na_correct = 0
        not_na_correct = 0
        for j in range(int(test_dataloader.entpair_tot / Config.batch_size)):
            batch_data = test_dataloader.next_batch()
            model.pos_word = to_tensor(batch_data["pos_word"])
            model.pos_pos1 = to_tensor(batch_data["pos_pos1"])
            model.pos_pos2 = to_tensor(batch_data["pos_pos2"])
            model.scope = batch_data["scope"]
            label = batch_data["label"]
            multi_label = batch_data["multi_label"]
            logit, output = model.test()
            logits.append(logit.cpu().detach().numpy().tolist())
            labels.append(multi_label.tolist())
            output = output.cpu().detach().numpy()
            tot += label.shape[0]
            tot_na += (label==0).sum()
            tot_not_na += (label!=0).sum()
            tot_correct += (label==output).sum()
            na_correct +=np.logical_and(label==output, label==0).sum()
            not_na_correct += np.logical_and(label==output, label!=0).sum()
            sys.stdout.write("dev:epoch:%d, acc:%.3f, na_acc:%.3f, not_na_acc:%.3f\r"%(i, tot_correct/tot, na_correct/tot_na, not_na_correct/tot_not_na))
            sys.stdout.flush()
        print("")
        eval(logits, labels)
        print("---------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--cuda", dest="cuda", type=str, default="7", help="cuda")
    parser.add_argument("--loss", dest="loss", type=str, default="ce", help="loss func")
    parser.add_argument("--neg_samples", dest="neg_samples",type=int, default=0, help="num of neg samples")
    args = parser.parse_args()
    
    # set para
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    Config.loss_func = args.loss
    Config.neg_samples = args.neg_samples

    # set save path
    if not os.path.exists(Config.save_path):
        os.mkdir(Config.save_path)
    
    # train
    train_dataloader = Dataloader("train")
    dev_dataloader = Dataloader("test")
    print(train_dataloader.weight)
    model = LatentRE(train_dataloader.word_vec, train_dataloader.weight)
    train(model, train_dataloader, dev_dataloader)

    # # test
    # if not os.path.exists(Config.save_path):
    #     exit("There are not checkpoints to test!")
    # train_dataloader = Dataloader("train")
    # test_dataloader = Dataloader("test")
    # model = LatentRE(train_dataloader.word_vec, train_dataloader.weight)


        

        


        
        

        
        
    