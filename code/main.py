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

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': Config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.lr, eps=Config.adam_epsilon, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=Config.warmup_steps, t_total=t_total)

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
            # inputs = {
            #     'input_ids':batch_data[0].cuda(),
            #     'attention_mask':batch_data[1].cuda(),
            #     'mask':batch_data[2].cuda(),
            #     'query':batch_data[3].cuda(),
            #     'knowledge':batch_data[4].cuda().float(),
            # } 
            inputs = {
                'pos_word':batch_data['pos_word'].cuda(),
                'pos_pos1':batch_data['pos_pos1'].cuda(),
                'pos_pos2':batch_data['pos_pos2'].cuda(),
                'pos_label':batch_data['pos_label'].cuda(),
                'pos_query':batch_data['pos_query'].cuda(),
                'pos_scope':batch_data['pos_scope'],
                'neg_word':batch_data['neg_word'].cuda(),
                'neg_pos1':batch_data['neg_pos1'].cuda(),
                'neg_pos2':batch_data['neg_pos2'].cuda(),
                'neg_label':batch_data['neg_label'].cuda(),
                'one_neg_label':batch_data['one_neg_label'].cuda(),
                'mul_label':batch_data['mul_label'].cuda(),
                'mul_num':batch_data['mul_num'].cuda(),
                'neg_scope':batch_data['neg_scope'],
            }
            # inputs = {
            #     'word':batch_data['word'].cuda(),
            #     'pos1':batch_data['pos1'].cuda(),
            #     'pos2':batch_data['pos2'].cuda(),
            #     'label':batch_data['label'].cuda(),
            #     'scope':batch_data['scope']
            # }

            loss = parallel_model(**inputs)
            loss = loss.mean()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            nn.utils.clip_grad_norm_(amp.master_params(optimizer), Config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            parallel_model.zero_grad()
            global_step += 1

            # output = output.cpu().detach().numpy()
            # label = batch_data[3].numpy()
            # tot += label.shape[0]
            # acc += (output == label).sum()
            # sys.stdout.write("epoch: %d, batch: %d, acc: %.3f, loss: %.6f\r" % (i, j, (acc/tot), loss))
            # sys.stdout.flush()

            sys.stdout.write("epoch: %d, batch: %d, loss: %.6f\r" % (i, j, loss))
            sys.stdout.flush()

            # final_input_words.append(batch_data[0].tolist())
            # final_mask_words.append(batch_data[2].tolist())
            # final_output_words.append(output.cpu().detach().numpy().tolist())

        print("")
        # clean gpu memory cache
        torch.cuda.empty_cache()
        
        # save model     
        if (i+1) % Config.save_epoch == 0:
            checkpoint = {
                # 'encoder': model.encoder.state_dict(),
                'selector': model.selector.state_dict(),
                'decoder': model.decoder.state_dict(),
            }
            torch.save(checkpoint, os.path.join(Config.save_path, "ckpt"+Config.info+str(i)))
            # json.dump(final_input_words, open(os.path.join("../output", Config.info+'input.json'), 'w'))
            # json.dump(final_mask_words, open(os.path.join("../output", Config.info+'mask.json'), 'w'))
            # json.dump(final_output_words, open(os.path.join("../output", Config.info+"output.json"), 'w'))
        
                # dev
        if (i+1) % Config.dev_step == 0:
            with torch.no_grad():
                print("begin deving...")
                parallel_model.eval()
                Config.training = False
                dev_iterator = (dev_ins_tot // Config.batch_size) if (dev_ins_tot % Config.batch_size == 0) else (dev_ins_tot // Config.batch_size + 1)
                for j in range(dev_iterator):
                    batch_data = dev_dataloader.next_batch()
                    # inputs = {
                    #     'input_ids':batch_data[0].cuda(),
                    #     'attention_mask':batch_data[1].cuda()
                    # }
                    inputs = {
                        'word':batch_data['word'].cuda(),
                        'pos1':batch_data['pos1'].cuda(),
                        'pos2':batch_data['pos2'].cuda(),
                    }
                    logit = parallel_model(**inputs)
                    bagTest.update(logit.cpu().detach())
                    sys.stdout.write("batch_size:%d, dev_ins_tot:%d, batch:%d, ,dev_processed: %.3f\r" % (Config.batch_size, dev_ins_tot, j, j/((dev_ins_tot // Config.batch_size))))
                    sys.stdout.flush()
                print("")
                bagTest.forward(i)  
                print("---------------------------------------------------------------------------------------------------")
                #clean gpu memory cache
                torch.cuda.empty_cache()
        
    # after iterator, save the best perfomance
    log(bagTest.auc, bagTest.epoch)

def test(model, test_dataloader, ins_tot):
    # just for bag test
    bagTest = BagTest(test_dataloader.entpair2scope, test_dataloader.data_query)
    model.cuda()
    print("begin testing...")
    Config.training = False
    for i in range(1, Config.max_epoch):
        # restore the stored params
        if not os.path.exists(os.path.join(Config.save_path, "ckpt"+str(i))):
            continue
        checkpoint = torch.load(os.path.join(Config.save_path, "ckpt"+str(i)))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        logits = []
        labels = []
        tot = 0
        tot_na = 0
        tot_not_na = 0
        tot_correct = 0
        na_correct = 0
        not_na_correct = 0
        test_iterator = (ins_tot // Config.batch_size) if (dev_ins_tot % Config.batch_size == 0) else (ins_tot // Config.batch_size + 1)
        for j in range(test_iterator):
            batch_data = test_dataloader.next_batch()
            inputs = {
                'input_ids':to_int_tensor(batch_data['input_ids']),
                'attention_mask':to_int_tensor(batch_data['attention_mask']),
                'scope':batch_data['scope']
            }
            logit = model(**inputs)
            bagTest.update(logit.cpu().detach())
            sys.stdout.write("test_processed: %.3f\r" % ((j+1) / test_iterator))
            sys.stdout.flush()
        bagTest.forward(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--cuda", dest="cuda", type=str, 
                        default="4", help="cuda")
    parser.add_argument("--batch_size", dest="batch_size", type=int, 
                        default=0, help="batch size")
    parser.add_argument("--dataset", dest="dataset", type=str,
                        default='nyt',help='dataset to use')
    parser.add_argument("--gen_loss_scale", dest="gen_loss_scale",type=float, 
                        default=1.0, help="loss scale for bert MLM")
    parser.add_argument("--kl_loss_scale", dest="kl_loss_scale",type=float, 
                        default=1.0, help="kl loss scale")
    parser.add_argument("--ce_loss_scale", dest="ce_loss_scale",type=float, 
                        default=1.0, help="ce loss scale")
    parser.add_argument("--lr", dest="lr", type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument("--info", dest="info",type=str, 
                        default="", help="info for model")


    parser.add_argument("--latent", action='store_true', 
                        help="Whether not to use label info")
    parser.add_argument("--mask_mode",dest="mask_mode",type=str, 
                        default="none",help="mask mode. you should select from {'none','entity','origin', 'governor'}")
    parser.add_argument("--mode", dest="mode",type=str, 
                        default="train", help="train or test")
    
    parser.add_argument("--train_bag", action='store_true', 
                        help="whether not to train on bag level")
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
    Config.gen_loss_scale = args.gen_loss_scale
    Config.kl_loss_scale = args.kl_loss_scale
    Config.ce_loss_scale = args.ce_loss_scale
    Config.lr = args.lr
    Config.latent = args.latent
    Config.mask_mode = args.mask_mode
    Config.train_bag = args.train_bag
    Config.eval_bag = args.eval_bag
    Config.max_epoch = args.max_epoch
    Config.dev_step = args.dev_step
    Config.save_epoch = args.save_epoch
    Config.dataset = args.dataset
    Config.seed = args.seed 
    print(args)

    # set save path
    if not os.path.exists(Config.save_path):
        os.mkdir(Config.save_path)
    # set seed
    set_seed(args)
    
    if args.mode == "train":
        # train
        train_dataloader = Dataloader('train', 'relfact' if Config.train_bag else 'ins', Config.dataset)
        dev_dataloader = Dataloader('test', 'entpair' if Config.eval_bag else 'ins', Config.dataset)
        model = LatentRE(train_dataloader.word_vec, train_dataloader.weight)
        model.cuda()
        train(args,
              model, 
              train_dataloader, 
              dev_dataloader, 
              train_dataloader.relfact_tot if Config.train_bag else train_dataloader.instance_tot,
              dev_dataloader.entpair_tot if Config.eval_bag else dev_dataloader.instance_tot)


        

        


        
        

        
        
    