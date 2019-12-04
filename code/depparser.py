
import os 
import json 
import pdb
import numpy as np 
from tqdm import trange
from config import Config
import sys 
import time
import argparse 
from transformers import BertTokenizer

class Depparser():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(Config.model_name_or_path, do_lower_case=True)
        pass 
    def parser(self, args):
        # merge depparsered data
        parser_tokened_word = json.load(open("../data/parser_tokened_word0.json"))
        word1 = json.load(open("../data/parser_tokened_word1.json"))
        parser_governor = json.load(open("../data/parser_governor0.json"))
        governor1 = json.load(open("../data/parser_governor1.json"))
        parser_tokened_word.extend(word1)
        parser_governor.extend(governor1)
        
        # origin_data
        data = json.load(open("../data/nyt/train.json"))
        data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + "#" + a["relation"])
        tot = len(data)
        if tot != len(parser_tokened_word):
            print("llll")
        # final mask
        mask_index = np.zeros((tot, Config.sen_len), dtype=int)
        # warning num 
        warning_num = 0
        warning_in_sen_num = 0 
        iter = trange(tot)
        for i in iter: 
            instance = data[i]
            sen = instance["sentence"].lower().replace(".", '')
            head = instance["head"]["word"].lower()
            tail = instance["tail"]["word"].lower()
            bert_tokens = self.tokenizer.tokenize(sen)
            bert_tokens.insert(0, '[CLS]')
            bert_tokens.append("[SEP]")
            tokened_word = parser_tokened_word[i]
            
            governor = parser_governor[i]
            mask = []
            head = head.split()[0]
            tail = tail.split()[0]
            len_head = len(head.split())
            len_tail = len(tail.split())
            try:
                hidx = tokened_word.index(head)
                tidx = tokened_word.index(tail)
            except:
                warning_num += 1
                continue
            head_path = []
            tail_path = []
            head_governor = governor[hidx]
            tail_governor = governor[tidx]

            # the first child
            child_head_path = []
            child_tail_path = []
            head_father = hidx+1
            tail_father = tidx+1
            for j, item in enumerate(governor):
                if item == head_father:
                    if j < hidx and j >= (hidx+len_head):
                        child_head_path.append(j+1)
                elif item == tail_father:
                    if j < tidx and j >= (tidx+len_tail):
                        child_tail_path.append(j+1)
            
            key_governor = -1
            while head_governor != 0:
                head_path.append(head_governor)
                head_governor = governor[head_governor-1]
            head_path.append(0)
            while tail_governor != 0:
                if tail_governor in head_path:
                    key_governor = tail_governor
                    break
                tail_path.append(tail_governor)
                tail_governor = governor[tail_governor-1]
            tail_path.append(0)
            if key_governor == -1:
                key_governor = 0
            for j in head_path:
                if j == key_governor:
                    mask.append(j)
                    break
                mask.append(j)
            for j in tail_path:
                mask.append(j)
            
            # child 
            for j in child_head_path:
                mask.append(j)
            for j in child_tail_path:
                mask.append(j)
            
            true_mask = [] 
            for j in mask:
                if j == 0:
                    continue
                else:
                    text = tokened_word[j-1]
                try:
                    id = bert_tokens.index(text)
                    true_mask.append(id)
                except:
                    warning_in_sen_num += 1
                    continue
            for m in true_mask:
                if m < Config.sen_len:
                    mask_index[i][m] = 1         
        print(warning_num / tot)
        print(warning_in_sen_num)
        np.save("../data/pre_processed_data/train_governor_mask_index.npy", mask_index)

parser = argparse.ArgumentParser(description="latentRE")
parser.add_argument("--cuda", dest="cuda", type=str, default="4", help="cuda")
parser.add_argument("--index", dest="index", type=int, default=0, help="index")
parser.add_argument("--child", action="store_true", help="whether or not to mask the first child")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda 
print(args)
depparser = Depparser()
depparser.parser(args)
        