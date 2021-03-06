"""
Train
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import sklearn.metrics
import matplotlib
import pdb
import numpy as np 
import time
import random
from apex import amp
from tqdm import tqdm
from tqdm import trange
import time
# Use 'Agg' so this program could run on a remote server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AdamW, WarmupLinearSchedule
from model import LatentRE
from config import Config
from dataloader import Dataloader, AdvDataloader
from dataloader import Dataset
from tmp_for_test import BagTest

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

def log_loss(epoch, loss):
    if not os.path.exists("../res"):
        os.mkdir("../res")
    f = open(os.path.join("../res", "loss"), "a+")
    f.write(time.asctime(time.localtime(time.time()))+"\n")
    f.write(Config.info+"\n")
    f.write("epoch:{}, loss:{}\n".format(epoch, loss))
    if epoch == Config.max_epoch - 1:
        f.write("---------------------------------------------------------------------------------------------------\n")
    f.close()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def train(args, model, train_dataloader, dev_dataloader, train_ins_tot, dev_ins_tot):
    # Prepare optimizer and schedule (linear warmup and decay)
    t_total = train_ins_tot // Config.batch_size * Config.max_epoch

    if Config.optimizer == "adamw":
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': Config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=Config.lr, eps=Config.adam_epsilon, correct_bias=False)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=Config.warmup_steps, t_total=t_total)
    elif Config.optimizer == "sgd":
        params = model.parameters()
        optimizer = optim.SGD(params, Config.lr)
    elif Config.optimizer == "adam":
        params = model.parameters()
        optimizer = optim.Adam(params, Config.lr)

    if Config.optimizer == "adamw":
        # amp training 
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # for bag test
    bagTest = BagTest(dev_dataloader.entpair2scope, dev_dataloader.data_query)

    # Data parallel
    parallel_model = nn.DataParallel(model)
    parallel_model.zero_grad()
   
    print("Begin train...")
    print("We will train model in %d steps"%(train_ins_tot//Config.batch_size//Config.gradient_accumulation_steps*Config.max_epoch))
    best_auc = 0
    best_epoch = 0
    global_step = 0
    set_seed(args)
    best_auc = 0
    for i in range(Config.max_epoch):
        # train
        acc = 0
        tot = 0
        final_input_words = []
        final_mask_words = []
        final_output_words = []
        parallel_model.train()
        Config.training = True
        # epoch_iterator = trange(int(train_ins_tot/Config.batch_size), desc="epoch "+str(i))
        for j in range(int(train_ins_tot/Config.batch_size)):
            batch_data = train_dataloader.next_batch()
            if Config.encoder == "bert":
                inputs = {
                    'input_ids':batch_data[0].cuda(),
                    'attention_mask':batch_data[1].cuda(),
                    'query':batch_data[2].cuda(),
                } 
            else:
                inputs = {
                    'word':batch_data[0].cuda(),
                    'pos1':batch_data[1].cuda(),
                    'pos2':batch_data[2].cuda(),
                    'pcnn_mask':batch_data[3].cuda(),
                    'label':batch_data[4].cuda(),
                    'query':batch_data[5].cuda(),
                    'scope':batch_data[6]
                }
            loss = parallel_model(**inputs)
            loss = loss.mean()
            if Config.optimizer == "adamw":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                nn.utils.clip_grad_norm_(amp.master_params(optimizer), Config.max_grad_norm)
            else:
                loss.backward()
            optimizer.step()
            if Config.optimizer == "adamw":
                scheduler.step()
            parallel_model.zero_grad()
            global_step += 1
            sys.stdout.write("epoch: %d, batch: %d, loss: %.6f\r" % (i, j, loss))
            sys.stdout.flush()
        print("")
        # clean gpu memory cache
        torch.cuda.empty_cache()
        
        # save model     
        # if (i+1) % Config.save_epoch == 0:

        
        # dev
        if (i+1) % Config.dev_step == 0:
            with torch.no_grad():
                print("begin deving...")
                parallel_model.eval()
                Config.training = False
                dev_iterator = (dev_ins_tot // Config.batch_size) if (dev_ins_tot % Config.batch_size == 0) else (dev_ins_tot // Config.batch_size + 1)
                for j in range(dev_iterator):
                    batch_data = dev_dataloader.next_batch()
                    inputs = {
                        'input_ids':batch_data[0].cuda(),
                        'attention_mask':batch_data[1].cuda(),
                        'word':batch_data[2].cuda(),
                        'pos1':batch_data[3].cuda(),
                        'pos2':batch_data[4].cuda(),
                        'pcnn_mask':batch_data[5].cuda(),
                        'scope':batch_data[6],
                    }
                    logit = parallel_model(**inputs)
                    bagTest.update(logit.cpu().detach())
                    sys.stdout.write("batch_size:%d, dev_ins_tot:%d, batch:%d, ,dev_processed: %.3f\r" % (Config.batch_size, dev_ins_tot, j, j/((dev_ins_tot // Config.batch_size))))
                    sys.stdout.flush()
                print("")
                auc = bagTest.forward(i)  
                print("---------------------------------------------------------------------------------------------------")
                #clean gpu memory cache
                torch.cuda.empty_cache()
        if  auc > best_auc:
            best_auc = auc
            checkpoint = {
                'model': model.state_dict(),
            }
            torch.save(checkpoint, os.path.join(Config.save_path, Config.info))
        
    # after iterator, save the best perfomance
    # log(bagTest.auc, bagTest.epoch)

def test(model, test_dataloader, ins_tot):
    # just for bag test
    bagTest = BagTest(test_dataloader.entpair2scope, test_dataloader.data_query)
    print("begin testing...")
    # restore the stored params
    checkpoint = torch.load(os.path.join(Config.save_path, Config.info))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    Config.training = False
    test_iterator = (ins_tot // Config.batch_size) if (ins_tot % Config.batch_size == 0) else (ins_tot // Config.batch_size + 1)
    with torch.no_grad():
        for j in range(test_iterator):
            batch_data = test_dataloader.next_batch()
            inputs = {
                'input_ids':batch_data[0].cuda(),
                'attention_mask':batch_data[1].cuda(),
                'word':batch_data[2].cuda(),
                'pos1':batch_data[3].cuda(),
                'pos2':batch_data[4].cuda(),
                'pcnn_mask':batch_data[5].cuda(),
                'scope':batch_data[6],
            }
            logit = model(**inputs)
            bagTest.update(logit.cpu().detach())
            sys.stdout.write("test_processed: %.3f\r" % ((j+1) / test_iterator))
            sys.stdout.flush()
        auc = bagTest.forward(0)
        log(auc, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--cuda", dest="cuda", type=str, 
                        default="4", help="cuda")
    parser.add_argument("--batch_size", dest="batch_size", type=int, 
                        default=0, help="batch size")
    parser.add_argument("--dataset", dest="dataset", type=str,
                        default='nyt',help='dataset to use')
    parser.add_argument("--lr", dest="lr", type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument("--info", dest="info",type=str, 
                        default="", help="info for model")
    parser.add_argument('--encoder', dest='encoder', type=str,
                        default='cnn', help='encoder type')
    parser.add_argument("--train_bag", action='store_true', 
                        help="whether not to train on bag level")
    parser.add_argument("--bag_type", dest="bag_type", type=str,
                        default='one',help='bag type')
    parser.add_argument("--hidden_size", dest="hidden_size", type=int,
                        default=768,help='hidden size')
    parser.add_argument("--optim", dest="optim", type=str,
                        default='sgd',help='optim type')
    parser.add_argument("--dump_logit", action='store_true', 
                        help="whether not to dump logit")
    

    parser.add_argument("--mode", dest="mode",type=str, 
                        default="train", help="train or test")
    parser.add_argument("--eval_bag", action='store_true', 
                        help="whether not to eval on bag level")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int, 
                        default=3, help="max epoch")
    parser.add_argument("--dev_step", dest="dev_step", type=int, 
                        default=1,help="dev epoch")
    parser.add_argument("--save_epoch", dest="save_epoch", type=int, 
                        default=100,help="save epoch")


    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")
    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")
    args = parser.parse_args()
    
    # set para
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    Config.info = args.info
    Config.batch_size = args.batch_size
    Config.lr = args.lr
    Config.hidden_size = args.hidden_size
    Config.encoder = args.encoder
    Config.optimizer = args.optim
    Config.train_bag = args.train_bag
    Config.bag_type = args.bag_type
    Config.num_feature = 3 if args.encoder == "pcnn" else 1
    Config.eval_bag = args.eval_bag
    Config.max_epoch = args.max_epoch
    Config.dev_step = args.dev_step 
    Config.save_epoch = args.save_epoch
    Config.dataset = args.dataset
    Config.seed = args.seed 
    Config.dump_logit = args.dump_logit
    print(args)

    # set save path
    if not os.path.exists(Config.save_path):
        os.mkdir(Config.save_path)
    # set seed
    set_seed(args)

    if args.mode == "train":
        # train
        train_dataloader = Dataloader('train', 'relfact' if Config.train_bag else 'ins', Config.dataset)
        dev_dataloader = Dataloader('dev', 'entpair' if Config.eval_bag else 'ins', Config.dataset)
        model = LatentRE(train_dataloader.word_vec, train_dataloader.weight)
        model.cuda()
        train(args,
              model, 
              train_dataloader, 
              dev_dataloader, 
              train_dataloader.relfact_tot if Config.train_bag else train_dataloader.instance_tot,
              dev_dataloader.entpair_tot if Config.eval_bag else dev_dataloader.instance_tot)

        # test
        os.system("rm -r ../data/pre_processed_data/*")
        test_dataloader = Dataloader('test', 'entpair' if Config.eval_bag else 'ins', Config.dataset)
        model = LatentRE(test_dataloader.word_vec, test_dataloader.weight)
        model.cuda()
        test(model,
             test_dataloader,
             test_dataloader.entpair_tot if Config.eval_bag else test_dataloader.instance_tot)
    elif args.mode == "test":
        test_dataloader = Dataloader('test', 'entpair' if Config.eval_bag else 'ins', Config.dataset)
        model = LatentRE(test_dataloader.word_vec, test_dataloader.weight)
        model.cuda()
        test(model,
             test_dataloader,
             test_dataloader.entpair_tot if Config.eval_bag else test_dataloader.instance_tot)




        

        


        
        

        
        
    