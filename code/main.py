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
import time
from apex import amp
# Use 'Agg' so this program could run on a remote server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AdamW, WarmupLinearSchedule
from model import LatentRE
from config import Config
from dataloader import Dataloader

def log(auc, step):
    if not os.path.exists("../res"):
        os.mkdir("../res")
    f = open(os.path.join("../res", "result"), 'a+')
    res = "AUC:{},\tepoch:{},\tlr:{}\n".format(auc, step, Config.lr)
    f.write(time.asctime(time.localtime(time.time()))+"\n")
    f.write(Config.info+"\n")
    f.write(res)
    f.write("---------------------------------------------------------------------------------------------------\n")
    f.close()



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
    print("auc = "+str(auc)+"| "+"F1 = "+str(f1))
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

    return auc

def to_int_tensor(array):
    return torch.from_numpy(array).to(torch.int64).cuda()

def to_float_tensor(array):
    return torch.from_numpy(array).to(torch.float32).cuda()

def train(model, train_dataloader, dev_dataloader=None):
    model.cuda()
    # params = filter(lambda x:x.requires_grad, model.parameters())
    # Prepare optimizer and schedule (linear warmup and decay)
    t_total = train_dataloader.instance_tot // Config.batch_size // Config.gradient_accumulation_steps * Config.max_epoch

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': Config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.lr, eps=Config.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=Config.warmup_steps, t_total=t_total)

    # amp training 
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # Data parallel
    model = nn.DataParallel(model)
    model.zero_grad()
    
    print("Begin train...")
    print("We will train model in %d steps"%(train_dataloader.entpair_tot//Config.batch_size//Config.gradient_accumulation_steps*Config.max_epoch))
    best_auc = 0
    best_epoch = 0
    global_step = 0
    for i in range(Config.max_epoch):
        model.train()
        Config.training = True
        tot = 0
        tot_na = 0
        tot_not_na = 0
        tot_correct = 0
        na_correct = 0
        not_na_correct = 0
        for j in range(int(train_dataloader.entpair_tot / Config.batch_size)):
            batch_data = train_dataloader.next_batch()            
            inputs = {
                'pos_word':to_int_tensor(batch_data['pos_word']),
                'pos_pos1':to_int_tensor(batch_data['pos_pos1']),
                'pos_pos2':to_int_tensor(batch_data['pos_pos2']),
                'knowledge':to_float_tensor(batch_data['knowledge']),
                'input_ids':to_int_tensor(batch_data['input_ids']),
                'attention_mask':to_int_tensor(batch_data['attention_mask'])
            }
            label = batch_data["label"]
            
            loss = model(**inputs)
            loss = loss.mean()
            loss = loss / Config.gradient_accumulation_steps
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            nn.utils.clip_grad_norm_(amp.master_params(optimizer), Config.max_grad_norm)
            
            # compute acc 
            # output = output.cpu().detach().numpy()
            # tot += label.shape[0]
            # tot_na += (label==0).sum()
            # tot_not_na += (label!=0).sum()
            # tot_correct += (label==output).sum()
            # na_correct +=np.logical_and(label==output, label==0).sum()
            # not_na_correct += np.logical_and(label==output, label!=0).sum()
            
            if (j+1) % Config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                # sys.stdout.write("train:epoch:%d, loss:%.3f, acc:%.3f, na_acc:%.3f, not_na_acc:%.3f\r"%(i, loss, tot_correct/tot, na_correct/tot_na, not_na_correct/tot_not_na))
                sys.stdout.write("train:epoch:%d, global_step:%d, loss:%.3f\r"%(i, global_step, loss))
                sys.stdout.flush()
        print("")
        
        if i % Config.dev_step == 0:
            print("begin deving...")
            model.eval()
            Config.training = False
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
                inputs = {
                    'pos_word':to_int_tensor(batch_data['pos_word']),
                    'pos_pos1':to_int_tensor(batch_data['pos_pos1']),
                    'pos_pos2':to_int_tensor(batch_data['pos_pos2']),
                    'scope':batch_data['scope']
                }
                                
                label = batch_data["label"]
                multi_label = batch_data["multi_label"]
                
                # compute auc 
                logit, output = model(**inputs)
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
            _auc = eval(logits, labels)
            if _auc > best_auc:
                best_auc = _auc
                best_epoch = i
            print("---------------------------------------------------------------------------------------------------")    
    log(best_auc, best_epoch) 
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
            inputs = {
                'pos_word':to_int_tensor(batch_data['pos_word']),
                'pos_pos1':to_int_tensor(batch_data['pos_pos1']),
                'pos_pos2':to_int_tensor(batch_data['pos_pos2']),
                'scope':batch_data['scope']
            }
            label = batch_data["label"]
            multi_label = batch_data["multi_label"]
            
            logit, output = model(**input)
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
    parser.add_argument("--cuda", dest="cuda", type=str, default="4", help="cuda")
    parser.add_argument("--loss", dest="loss", type=str, default="ce", help="loss func")
    parser.add_argument("--neg_samples", dest="neg_samples",type=int, default=1, help="num of neg samples")
    parser.add_argument("--info", dest="info",type=str, default="", help="info for model")
    args = parser.parse_args()
    
    # set para
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    Config.loss_func = args.loss
    Config.neg_samples = args.neg_samples
    Config.info = args.info

    # set save path
    if not os.path.exists(Config.save_path):
        os.mkdir(Config.save_path)
    if not os.path.exists("../visualizing"):
        os.mkdir("../visualizing")
    
    # train
    train_dataloader = Dataloader("train", "ins")
    dev_dataloader = Dataloader("test", "entpair")
    print(train_dataloader.weight)
    model = LatentRE(train_dataloader.word_vec, train_dataloader.weight)
    train(model, train_dataloader, dev_dataloader)

    # # test
    # if not os.path.exists(Config.save_path):
    #     exit("There are not checkpoints to test!")
    # train_dataloader = Dataloader("train")
    # test_dataloader = Dataloader("test")
    # model = LatentRE(train_dataloader.word_vec, train_dataloader.weight)


        

        


        
        

        
        
    