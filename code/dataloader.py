"""
# This file will pre-process data to .npy format.
# And this will be convenient for model to get data
"""
import os 
import sys
import json 
import numpy as np 
import random
import pdb
import time
from config import Config
from transformers import BertTokenizer, GPT2Tokenizer
import multiprocessing.dummy as mp 
from torch import utils
import torch

class Dataset(utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels, query, knowledge, length):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.query = query
        self.knowledge = knowledge
        self.length = length
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        length = self.length[index]
        input_ids = self.input_ids[index]
        atttention_mask = self.attention_mask[index]
        labels = self.labels[index]
        query = self.query[index]
        knowledge = self.knowledge[index]

        return input_ids, atttention_mask, labels, query, knowledge, length

class Dataloader:
    '''
    # This class 
    '''
    def __init__(self, mode, flag, dataset):    
        np.random.seed(Config.seed)
        self.mode = mode
        if not os.path.exists("../data/pre_processed_data"):
            os.mkdir("../data/pre_processed_data")
        if not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_label.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_length.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_knowledge.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_entpair2scope.json")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_relfact2scope.json")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_input_ids.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_attention_mask.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_token_mask.npy")):
        # not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_word.npy")):
            print("There dones't exist pre-processed data, pre-processing...")
            start_time = time.time()
            data = json.load(open(os.path.join("../data/"+dataset, mode+".json")))
            knowledge = json.load(open(os.path.join("../data/knowledge",dataset+"_"+mode+'.json')))
            ori_word_vec = json.load(open(os.path.join("../data/"+dataset,"word_vec.json")))
            Config.word_tot = len(ori_word_vec) + 2

            # Bert tokenizer
            bert_tokenizer = BertTokenizer.from_pretrained(Config.model_name_or_path, do_lower_case=True)
            gpt2_tokenizer = GPT2Tokenizer.from_pretrained(Config.gpt2, do_lower_case=True)

            # process word vec
            word2id = {}
            word2id["blk"] = 0
            word2id["unk"] = 1
            for word in ori_word_vec:
                w = word["word"].lower()
                word2id[w] = len(word2id)
            
            # process rel2id
            rel2id = json.load(open(os.path.join("../data/"+dataset,"rel2id.json")))
            Config.rel_num = len(rel2id)

            # process word_vec
            word_vec = []
            word_vec.append(np.zeros((len(ori_word_vec[0]["vec"]))))
            word_vec.append(np.random.random_sample(len(ori_word_vec[0]["vec"])))
            for word in ori_word_vec:
                word_vec.append(word["vec"])
            self.word_vec = np.asarray(word_vec)
            Config.word_embeeding_dim = len(word_vec[0])
            
            # sort data by head and tail and get entities-pos dict          
            data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + "#" + a["relation"])   
            entities_pos_dict = {}    
            relfact2scope = {}
            curr_entities = data[0]["head"]["id"]+"#"+data[0]["tail"]["id"]
            curr_relfact = data[0]["head"]["id"]+"#"+data[0]["tail"]["id"]+"#"+data[0]["relation"]
            entities_pos_dict[curr_entities] = [0,] 
            relfact2scope[curr_relfact] = [0,]
            for i, instance in enumerate(data):
                entities = instance["head"]["id"]+"#"+instance["tail"]["id"]
                relfact = instance["head"]["id"]+"#"+instance["tail"]["id"] + "#"+instance["relation"]
                if entities!=curr_entities:
                    entities_pos_dict[curr_entities].append(i)
                    curr_entities = entities
                    entities_pos_dict[curr_entities] = [i,]
                if relfact!=curr_relfact:
                    relfact2scope[curr_relfact].append(i)
                    curr_relfact = relfact
                    relfact2scope[curr_relfact] = [i,]
            entities_pos_dict[curr_entities].append(len(data))
            relfact2scope[curr_relfact].append(len(data))
            self.entpair2scope = entities_pos_dict
            self.relfact2scope = relfact2scope

            # process data
            self.instance_tot = len(data)
            self.data_length = np.zeros((self.instance_tot, ), dtype=int)
            self.data_decoder_length = np.zeros((self.instance_tot, ), dtype=int)
            self.data_query = np.zeros((self.instance_tot, ), dtype=int)
            self.data_knowledge = np.zeros((self.instance_tot, Config.rel_num), dtype=float)
            # for encoder
            self.data_input_ids = np.zeros((self.instance_tot, Config.sen_len), dtype=int)
            self.data_attention_mask = np.zeros((self.instance_tot, Config.sen_len), dtype=int)
            # for decoder
            self.data_decoder_input_ids = np.zeros((self.instance_tot, Config.sen_len), dtype=int)
            self.data_decoder_attention_mask = np.zeros((self.instance_tot, Config.sen_len), dtype=int)
            self.data_token_mask = np.ones((self.instance_tot, Config.sen_len), dtype=int)
            self.data_between_entity_mask = np.zeros((self.instance_tot, Config.sen_len), dtype=int)
            self.data_labels = np.zeros((self.instance_tot, Config.sen_len), dtype=int) - 1 # -1

            def _process_loop(i):
                instance = data[i]
                head = instance["head"]["word"].lower()
                tail = instance["tail"]["word"].lower()
                sentence = instance["sentence"].lower()
                try:
                    self.data_query[i] = rel2id[instance["relation"]]
                except:
                    self.data_query[i] = 0
                    print("relation error 1")
                
                # for bert encoder
                bert_tokens = bert_tokenizer.tokenize(sentence)
                head_tokens = bert_tokenizer.tokenize(head)
                tail_tokens = bert_tokenizer.tokenize(tail)
                head_pos = bert_tokens.index(head_tokens[0])
                bert_tokens.insert(head_pos, "[unused0]")
                bert_tokens.insert(head_pos+len(head_tokens)+1, "[unused1]")
                tail_pos = bert_tokens.index(tail_tokens[0])
                bert_tokens.insert(tail_pos, "[unused2]")
                bert_tokens.insert(tail_pos+len(tail_tokens)+1, "[unused3]")
                bert_tokens.insert(0, "[CLS]")
                bert_tokens.append("[SEP]")
                length = min(len(bert_tokens), Config.sen_len)
                self.data_input_ids[i][0:length] = bert_tokenizer.convert_tokens_to_ids(bert_tokens[0:length])
                self.data_attention_mask[i][0:length] = 1
                self.data_length[i] = length                
                
                # for mask 
                # words = sentence.split()
                # head_tokens = head.split()
                # tail_tokens = tail.split()
                # head_pos = words.index(head_tokens[0])
                # words.insert(head_pos, "#")
                # words.insert(head_pos+len(head_tokens)+1, "*")
                # tail_pos = words.index(tail_tokens[0])
                # words.insert(tail_pos, "^")
                # words.insert(tail_pos+len(tail_tokens)+1, "`")
                # sentence = ''
                # for word in words:
                #     sentence += word
                #     sentence += ' '
                # gpt2_tokens = gpt2_tokenizer.tokenize(sentence)
                # # try:
                # token1 = "Ġ#"
                # token2 = "Ġ*"
                # token3 = "Ġ^"
                # token4 = "Ġ`"
                # try:
                #     gpt2_tokens.index(token1)
                # except:
                #     token1 = "#"
                # try:
                #     gpt2_tokens.index(token2)
                # except:
                #     token2 = '*'
                # try:
                #     gpt2_tokens.index(token3)
                # except:
                #     token3 = '^'
                # try:
                #     gpt2_tokens.index(token4)
                # except:
                #     token4 = '`'
                # head_pos = gpt2_tokens.index(token1)
                # tail_pos = gpt2_tokens.index(token3)
                # if head_pos < tail_pos:
                #     len_head = gpt2_tokens.index(token2) - 1 - head_pos
                #     gpt2_tokens.remove(token1)
                #     gpt2_tokens.remove(token2)
                #     tail_pos = gpt2_tokens.index(token3)
                #     len_tail = gpt2_tokens.index(token4) - 1 - tail_pos
                #     gpt2_tokens.remove(token3)
                #     gpt2_tokens.remove(token4)
                # else:
                #     len_tail = gpt2_tokens.index(token4) - 1 - tail_pos
                #     gpt2_tokens.remove(token3)
                #     gpt2_tokens.remove(token4)
                #     head_pos = gpt2_tokens.index(token1)
                #     len_head = gpt2_tokens.index(token2) - 1 - head_pos
                #     gpt2_tokens.remove(token1)
                #     gpt2_tokens.remove(token2)
                # length = min(len(gpt2_tokens), Config.sen_len)
                # if head_pos < tail_pos:
                #     fir_pos = head_pos
                #     sec_pos = tail_pos
                #     len_fir = len_head
                # else:
                #     fir_pos = tail_pos
                #     sec_pos = head_pos
                #     len_fir = len_tail
                # gpt2_tokens_final = []
                # gpt2_tokens_final.extend(gpt2_tokens[sec_pos:length])
                # gpt2_tokens_final.extend(gpt2_tokens[0:sec_pos])
                # self.data_decoder_input_ids[i][0:length] = gpt2_tokenizer.convert_tokens_to_ids(gpt2_tokens_final[0:length])
                # self.data_decoder_attention_mask[i][0:length] = 1
                # self.data_between_entity_mask[i][fir_pos+len_fir+length-sec_pos:length] = 1
                sentence = head + ' ' + '*' + ' ' + tail
                gpt2_tokens = gpt2_tokenizer.tokenize(sentence)
                rel_pos = gpt2_tokens.index('Ġ*')
                length = min(len(gpt2_tokens), Config.sen_len)
                self.data_decoder_input_ids[i][0:length] = gpt2_tokenizer.convert_tokens_to_ids(gpt2_tokens[0:length])
                self.data_decoder_attention_mask[i][0:length] = 1
                self.data_between_entity_mask[i][rel_pos] = 1
                self.data_labels[i][rel_pos+1:length] = self.data_decoder_input_ids[i][rel_pos+1:length]
                self.data_decoder_length[i] = length

                # knowledge 
                entities = instance["head"]["id"]+"#"+instance["tail"]["id"]
                rels = knowledge[entities]
                rel_num = len(rels)
                for rel in rels:
                    try:
                        self.data_knowledge[i][rel2id[rel]] = 1 / rel_num
                    except:
                        self.data_knowledge[i][0] += 1 / rel_num
                        print("relation error 2")

            print("begin multiple thread processing...")
            pool = mp.Pool(40)
            pool.map(_process_loop, range(0, self.instance_tot))

            # save array
            np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_query.npy"), self.data_query)
            np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_length.npy"), self.data_length)
            np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_decoder_length.npy"), self.data_decoder_length)
            np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_knowledge.npy"), self.data_knowledge)
            np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_input_ids.npy"), self.data_input_ids)
            np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_attention_mask.npy"), self.data_attention_mask)
            np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_decoder_input_ids.npy"), self.data_decoder_input_ids)
            np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_decoder_attention_mask.npy"), self.data_decoder_attention_mask)
            np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_token_mask.npy"), self.data_token_mask)
            np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_between_entity_mask.npy"), self.data_between_entity_mask) 
            np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_labels.npy"), self.data_labels) 
            json.dump(self.entpair2scope, open(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_entpair2scope.json"), 'w'))
            json.dump(self.relfact2scope, open(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_relfact2scope.json"), "w"))
            print("end pre-process")
            end_time = time.time()
            print(end_time-start_time)
        else:
            print("There exists pre-processed data already. loading....")
            self.word_vec = None
            self.data_query = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_query.npy"))
            self.data_length = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_length.npy"))
            self.data_decoder_length = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_decoder_length.npy"))
            self.data_knowledge = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_knowledge.npy"))
            self.data_input_ids = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_input_ids.npy"))
            self.data_attention_mask = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_attention_mask.npy"))
            self.data_decoder_input_ids = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_decoder_input_ids.npy"))
            self.data_decoder_attention_mask = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_decoder_attention_mask.npy"))
            self.data_labels = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_labels.npy"))
            self.entpair2scope = json.load(open(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_entpair2scope.json")))
            self.relfact2scope = json.load(open(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_relfact2scope.json")))
            Config.rel_num = len(json.load(open(os.path.join("../data/"+dataset, "rel2id.json"))))
            print("Finish loading...")
            self.instance_tot = self.data_input_ids.shape[0]
        self.entpair_tot = len(self.entpair2scope)
        self.relfact_tot = len(self.relfact2scope)

        # mask mode 
        if self.mode == "train":
            if Config.mask_mode == "entity":
                self.data_mask = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_token_mask.npy"))
            elif Config.mask_mode == "between":
                self.data_mask = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_between_entity_mask.npy"))
            elif Config.mask_mode == "origin":
                self.data_mask = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_token_mask.npy"))
            elif Config.mask_mode == "governor":
                self.data_mask = np.load(os.path.join("../data/pre_processed_data", mode+"_governor_mask_index.npy"))
            else:
                self.data_mask = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_token_mask.npy"))
        
        # order to train and scope to train
        self.flag = flag
        self.scope = []
        if flag == "entpair":
            self.order = list(range(self.entpair_tot))
            for key in self.entpair2scope.keys():
                self.scope.append(self.entpair2scope[key])
        elif flag == "relfact":
            self.order = list(range(self.relfact_tot))
            for key in self.relfact2scope.keys():
                self.scope.append(self.relfact2scope[key])
        elif flag == "ins":
            self.order = list(range(self.instance_tot))
        self.idx = 0
        # weight for train crossEntropyloss
        self.weight = np.zeros((Config.rel_num), dtype=float)
        for i in self.data_query:
            self.weight[i] += 1
        self.weight = 1 / self.weight**0.05

    def to_tensor(self, array):
        return torch.from_numpy(array)

    def next_batch(self):
        if self.idx >= len(self.order):
            if Config.training:
                random.shuffle(self.order)
            self.idx = 0
        idx0 = self.idx
        idx1 = self.idx + Config.batch_size
        if idx1 > len(self.order):
            idx1 = len(self.order)
        self.idx = idx1
        if self.flag == "ins":
            index = self.order[idx0:idx1]
            max_length = self.data_length[index].max()
            max_decoder_length = self.data_decoder_length[index].max()
            if Config.training:
                return self.to_tensor(self.data_input_ids[index][:, :max_length]), \
                        self.to_tensor(self.data_attention_mask[index][:, :max_length]), \
                         self.to_tensor(self.data_mask[index][:, :max_decoder_length]), \
                          self.to_tensor(self.data_query[index]), \
                           self.to_tensor(self.data_knowledge[index]), \
                            self.to_tensor(self.data_decoder_input_ids[index][:, :max_decoder_length]), \
                             self.to_tensor(self.data_decoder_attention_mask[index][:, :max_decoder_length]), \
                              self.to_tensor(self.data_labels[index][:, :max_decoder_length])
            else:
                return self.to_tensor(self.data_input_ids[index][:, :max_length]), \
                        self.to_tensor(self.data_attention_mask[index][:, :max_length])
        else:
            batch_data = {}
            _word = []
            _pos1 = []
            _pos2 = []
            _ids = []
            _mask = []
            _rel = []
            _label = []
            _multi_rel = []
            _scope = []
            cur_pos = 0
            for i in range(idx0, idx1):
                _word.append(self.data_word[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _pos1.append(self.data_pos1[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _pos2.append(self.data_pos2[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _ids.append(self.data_input_ids[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _mask.append(self.data_attention_mask[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _rel.append(self.data_query[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _label.append(self.data_query[self.scope[self.order[i]][0]])
                bag_size = self.scope[self.order[i]][1] - self.scope[self.order[i]][0]
                _scope.append([cur_pos, cur_pos + bag_size])
                cur_pos = cur_pos + bag_size
                _one_multi_rel = np.zeros((Config.rel_num), dtype=np.int32)
                for j in range(self.scope[self.order[i]][0], self.scope[self.order[i]][1]):
                    _one_multi_rel[self.data_query[j]] = 1
                _multi_rel.append(_one_multi_rel)
            
            batch_data['word'] = self.to_tensor(np.concatenate(_word))
            batch_data['pos1'] = self.to_tensor(np.concatenate(_pos1))
            batch_data['pos2'] = self.to_tensor(np.concatenate(_pos2))
            batch_data['query'] = self.to_tensor(np.concatenate(_rel))
            batch_data['label'] = self.to_tensor(np.stack(_label))
            batch_data['scope'] = np.stack(_scope)

            return batch_data


            


        

    
            




        
        
