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
        if not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_query.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_length.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_knowledge.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_entpair2scope.json")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_relfact2scope.json")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_input_ids.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_attention_mask.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_token_mask.npy")):
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
            # self.data_word = np.zeros((self.instance_tot, Config.sen_len), dtype=int)
            # self.data_pos1 = np.zeros((self.instance_tot, Config.sen_len), dtype=int)
            # self.data_pos2 = np.zeros((self.instance_tot, Config.sen_len), dtype=int)
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

                
                # # for cnn
                # p1 = sentence.find(' ' + head + ' ')
                # p2 = sentence.find(' ' + tail + ' ')
                # if p1 == -1:
                #     if sentence[:len(head) + 1] == head + " ":
                #         p1 = 0
                #     elif sentence[-len(head) - 1:] == " " + head:
                #         p1 = len(sentence) - len(head)
                #     else:
                #         p1 = 0 # shouldn't happen
                # else:
                #     p1 += 1
                # if p2 == -1:
                #     if sentence[:len(tail) + 1] == tail + " ":
                #         p2 = 0
                #     elif sentence[-len(tail) - 1:] == " " + tail:
                #         p2 = len(sentence) - len(tail)
                #     else:
                #         p2 = 0 # shouldn't happen
                # else:
                #     p2 += 1
                # words = sentence.split()
                # cur_ref_data_word = self.data_word[i]         
                # cur_pos = 0
                # pos1 = -1
                # pos2 = -1
                # for j, word in enumerate(words):
                #     if j < Config.sen_len:
                #         word = word.lower()
                #         if word in word2id:
                #             cur_ref_data_word[j] = word2id[word]
                #         else:
                #             cur_ref_data_word[j] = 1
                #     if cur_pos == p1:
                #         pos1 = j
                #         p1 = -1
                #     if cur_pos == p2:
                #         pos2 = j
                #         p2 = -1
                #     cur_pos += len(word) + 1
                # for j in range(j + 1, Config.sen_len):
                #     cur_ref_data_word[j] = 0
                # if pos1 == -1 or pos2 == -1:
                #     raise Exception("[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sentence, head, tail))
                # if pos1 >= Config.sen_len:
                #     pos1 = Config.sen_len - 1
                # if pos2 >= Config.sen_len:
                #     pos2 = Config.sen_len - 1
                # for j in range(Config.sen_len):
                #     self.data_pos1[i][j] = j - pos1 + Config.sen_len
                #     self.data_pos2[i][j] = j - pos2 + Config.sen_len


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
                
                head_pos = bert_tokens.index(head_tokens[0])
                tail_pos = bert_tokens.index(tail_tokens[0])
                if head_pos < tail_pos:
                    fir_pos = head_pos
                    sec_pos = tail_pos
                    len_fir = len(head_tokens)
                    len_sec = len(tail_tokens)
                else:
                    fir_pos = tail_pos
                    sec_pos = head_pos
                    len_fir = len(tail_tokens)
                    len_sec = len(head_tokens)
                self.data_between_entity_mask[i][fir_pos+len_fir+1:sec_pos-1] = 1     
                self.data_token_mask[i][head_pos-1:head_pos+len(head_tokens)+1] = 0
                self.data_token_mask[i][tail_pos-1:tail_pos+len(tail_tokens)+1] = 0

                # mask not entity
                # self.data_input_ids[i][head_pos-1:head_pos+len(head_tokens)+1] = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.mask_token)   
                # self.data_input_ids[i][tail_pos-1:tail_pos+len(tail_tokens)+1] = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.mask_token)

                self.data_input_ids[i][0:fir_pos-1] = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.mask_token)   
                self.data_input_ids[i][fir_pos+len_fir+1:sec_pos-1] = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.mask_token)
                self.data_input_ids[i][sec_pos+len_sec+1:length] = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.mask_token) 

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
                # # self.data_between_entity_mask[i][fir_pos+len_fir+length-sec_pos:length] = 1
                # self.data_labels[i][fir_pos+len_fir+length-sec_pos:length] = self.data_decoder_input_ids[i][fir_pos+len_fir+length-sec_pos:length]
                # self.data_decoder_length[i] = length
                
                # sentence = head + ' ' + '*' + ' ' + tail
                # gpt2_tokens = gpt2_tokenizer.tokenize(sentence)
                # rel_pos = gpt2_tokens.index('Ġ*')
                # length = min(len(gpt2_tokens), Config.sen_len)
                # self.data_decoder_input_ids[i][0:length] = gpt2_tokenizer.convert_tokens_to_ids(gpt2_tokens[0:length])
                # self.data_decoder_attention_mask[i][0:length] = 1
                # self.data_between_entity_mask[i][rel_pos] = 1
                # self.data_labels[i][rel_pos+1:length] = self.data_decoder_input_ids[i][rel_pos+1:length]
                # self.data_decoder_length[i] = length


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
            # np.save(os.path.join("../data/pre_processed_data", "word_vec.npy"), self.word_vec)
            # np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_word.npy"), self.data_word)
            # np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_pos1.npy"), self.data_pos1)
            # np.save(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_pos2.npy"), self.data_pos2)
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
            # self.word_vec = np.load(os.path.join("../data/pre_processed_data", "word_vec.npy"))
            # self.data_word = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_word.npy"))
            # self.data_pos1 = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_pos1.npy"))
            # self.data_pos2 = np.load(os.path.join("../data/pre_processed_data", dataset+"_"+mode+"_pos2.npy"))
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
            # Config.word_tot = len(self.word_vec) + 2
            # Config.word_embeeding_dim = len(self.word_vec[0])
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
            if Config.training:
                return self.to_tensor(self.data_input_ids[index][:, :max_length]), \
                        self.to_tensor(self.data_attention_mask[index][:, :max_length]), \
                         self.to_tensor(self.data_mask[index][:, :max_length]), \
                          self.to_tensor(self.data_query[index]), \
                           self.to_tensor(self.data_knowledge[index])
                            # self.to_tensor(self.data_decoder_input_ids[index][:, :max_length]), \
                            #  self.to_tensor(self.data_decoder_attention_mask[index][:, :max_length])
            else:
                return self.to_tensor(self.data_input_ids[index][:, :max_length]), \
                        self.to_tensor(self.data_attention_mask[index][:, :max_length])
            # batch_data = {}
            # batch_data["word"] = self.to_tensor(self.data_word[index])
            # batch_data["pos1"] = self.to_tensor(self.data_pos1[index])
            # batch_data["pos2"] = self.to_tensor(self.data_pos2[index])
            # return batch_data
        else:
            batch_data = {}
            _word = []
            _pos1 = []
            _pos2 = []
            _rel = []
            _label = []
            _multi_rel = []
            _scope = []
            cur_pos = 0
            for i in range(idx0, idx1):
                _word.append(self.data_word[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _pos1.append(self.data_pos1[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _pos2.append(self.data_pos2[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
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

class AdvDataloader:
    def __init__(self, mode):
        if mode == 'train':
            if not os.path.exists("../data/nyt/postive_train.json"):
                self.genBag()
            if not os.path.exists("../data/pre_processed_data/train_positive_word.npy"):
                print("begin pre processing train data")
                positive_train = json.load(open("../data/nyt/postive_train.json"))
                negative_train = json.load(open("../data/nyt/negative_train.json"))
                negative_relfact2rel = json.load(open("../data/nyt/negative_relfact2rel.json"))
                negative_relfact2scope = json.load(open("../data/nyt/negative_relfact2scope.json"))
                
                ori_word_vec = json.load(open(os.path.join("../data/nyt","word_vec.json")))
                Config.word_tot = len(ori_word_vec) + 2
                
                # process word vec
                word2id = {}
                word2id["blk"] = 0
                word2id["unk"] = 1
                for word in ori_word_vec:
                    w = word["word"].lower()
                    word2id[w] = len(word2id)
                
                # process rel2id
                rel2id = json.load(open(os.path.join("../data/nyt","rel2id.json")))
                Config.rel_num = len(rel2id)

                # process word_vec
                word_vec = []
                word_vec.append(np.zeros((len(ori_word_vec[0]["vec"]))))
                word_vec.append(np.random.random_sample(len(ori_word_vec[0]["vec"])))
                for word in ori_word_vec:
                    word_vec.append(word["vec"])
                word_vec = np.asarray(word_vec)
                Config.word_embeeding_dim = len(word_vec[0])

                np.save(os.path.join("../data/pre_processed_data", "word_vec.npy"), word_vec)
                self.convert(positive_train, word2id, rel2id, 'positive')
                self.convert(negative_train, word2id, rel2id, 'negative')

                data_multi_query = np.zeros((len(negative_train), Config.rel_num), dtype=int)
                for key in negative_relfact2scope.keys():
                    # pdb.set_trace()
                    scope = negative_relfact2scope[key]
                    ori_rels = negative_relfact2rel[key]
                    rels = []
                    for rel in ori_rels:
                        try:
                            rels.append(rel2id[rel])
                        except:
                            rels.append(0)
                    for i in range(scope[0], scope[1]):
                        data_multi_query[i][rels] = 1
                np.save(os.path.join("../data/pre_processed_data", "train_multi_query.npy"), data_multi_query)

                pos_one_query = np.zeros((len(positive_train),), dtype=int)
                for i, ins in enumerate(positive_train):
                    try:
                        pos_one_query[i] = rel2id[ins['relation']]
                    except:
                        pos_one_query[i] = 0
                np.save(os.path.join("../data/pre_processed_data", "train_one_query.npy"), pos_one_query)


            print("loading")                
            self.train_positive_word = np.load(os.path.join("../data/pre_processed_data", "train_positive_word.npy"))
            self.train_positive_pos1 = np.load(os.path.join("../data/pre_processed_data", "train_positive_pos1.npy"))
            self.train_positive_pos2 = np.load(os.path.join("../data/pre_processed_data", "train_positive_pos2.npy"))
            self.train_positive_query = np.load(os.path.join("../data/pre_processed_data", "train_positive_query.npy"))
            self.train_negative_word = np.load(os.path.join("../data/pre_processed_data", "train_negative_word.npy"))
            self.train_negative_pos1 = np.load(os.path.join("../data/pre_processed_data", "train_negative_pos1.npy"))
            self.train_negative_pos2 = np.load(os.path.join("../data/pre_processed_data", "train_negative_pos2.npy"))
            self.train_negative_query = np.load(os.path.join("../data/pre_processed_data", "train_negative_query.npy"))
            self.train_one_query = np.load(os.path.join("../data/pre_processed_data", "train_one_query.npy"))
            self.train_multi_query = np.load(os.path.join("../data/pre_processed_data", "train_multi_query.npy"))
            self.positive_relfact2scope = json.load(open(os.path.join("../data/nyt/positive_relfact2scope.json")))
            self.negative_relfact2scope = json.load(open(os.path.join("../data/nyt/negative_relfact2scope.json")))
            self.word_vec = np.load(os.path.join("../data/pre_processed_data", "word_vec.npy"))
            Config.rel_num = len(json.load(open(os.path.join("../data/nyt", "rel2id.json"))))
            Config.word_tot = len(self.word_vec) + 2
            Config.word_embeeding_dim = len(self.word_vec[0])

            if len(self.positive_relfact2scope) != len(self.negative_relfact2scope):
                exit("error!!!!")
            self.relfact_tot = len(self.positive_relfact2scope)
            self.order = list(range(self.relfact_tot))
            self.positive_scope = []
            for key in self.positive_relfact2scope.keys():
                self.positive_scope.append(self.positive_relfact2scope[key])
            self.negative_scope = []
            for key in self.negative_relfact2scope.keys():
                self.negative_scope.append(self.negative_relfact2scope[key])
            self.idx = 0
        
        elif mode == "test":
            if not os.path.exists("../data/pre_processed_data/test_word.npy"):
                print("pre processing test data")
                data = json.load(open('../data/nyt/test.json'))
                ori_word_vec = json.load(open(os.path.join("../data/nyt","word_vec.json")))
                Config.word_tot = len(ori_word_vec) + 2
                
                # process word vec
                word2id = {}
                word2id["blk"] = 0
                word2id["unk"] = 1
                for word in ori_word_vec:
                    w = word["word"].lower()
                    word2id[w] = len(word2id)
                
                # process rel2id
                rel2id = json.load(open(os.path.join("../data/nyt","rel2id.json")))
                Config.rel_num = len(rel2id)

                # process word_vec
                word_vec = []
                word_vec.append(np.zeros((len(ori_word_vec[0]["vec"]))))
                word_vec.append(np.random.random_sample(len(ori_word_vec[0]["vec"])))
                for word in ori_word_vec:
                    word_vec.append(word["vec"])
                word_vec = np.asarray(word_vec)
                Config.word_embeeding_dim = len(word_vec[0])

                # sort data by head and tail and get entities-pos dict          
                data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + "#" + a["relation"])   
                entities_pos_dict = {}    
                curr_entities = data[0]["head"]["id"]+"#"+data[0]["tail"]["id"]
                entities_pos_dict[curr_entities] = [0,] 
                for i, instance in enumerate(data):
                    entities = instance["head"]["id"]+"#"+instance["tail"]["id"]
                    if entities!=curr_entities:
                        entities_pos_dict[curr_entities].append(i)
                        curr_entities = entities
                        entities_pos_dict[curr_entities] = [i,]
                entities_pos_dict[curr_entities].append(len(data))

                instance_tot = len(data)
                data_word = np.zeros((instance_tot, Config.sen_len), dtype=int)
                data_pos1 = np.zeros((instance_tot, Config.sen_len), dtype=int)
                data_pos2 = np.zeros((instance_tot, Config.sen_len), dtype=int)
                data_query = np.zeros((instance_tot, Config.rel_num), dtype=int)

                for i in range(len(data)):
                    instance = data[i]
                    head = instance["head"]["word"].lower()
                    tail = instance["tail"]["word"].lower()
                    sentence = instance["sentence"].lower()
                    try:
                        data_query[i] = rel2id[instance["relation"]]
                    except:
                        data_query[i] = 0
                        print("relation error 1")

                    p1 = sentence.find(' ' + head + ' ')
                    p2 = sentence.find(' ' + tail + ' ')
                    if p1 == -1:
                        if sentence[:len(head) + 1] == head + " ":
                            p1 = 0
                        elif sentence[-len(head) - 1:] == " " + head:
                            p1 = len(sentence) - len(head)
                        else:
                            p1 = 0 # shouldn't happen
                    else:
                        p1 += 1
                    if p2 == -1:
                        if sentence[:len(tail) + 1] == tail + " ":
                            p2 = 0
                        elif sentence[-len(tail) - 1:] == " " + tail:
                            p2 = len(sentence) - len(tail)
                        else:
                            p2 = 0 # shouldn't happen
                    else:
                        p2 += 1
                    words = sentence.split()
                    cur_ref_data_word = data_word[i]         
                    cur_pos = 0
                    pos1 = -1
                    pos2 = -1
                    for j, word in enumerate(words):
                        if j < Config.sen_len:
                            word = word.lower()
                            if word in word2id:
                                cur_ref_data_word[j] = word2id[word]
                            else:
                                cur_ref_data_word[j] = 1
                        if cur_pos == p1:
                            pos1 = j
                            p1 = -1
                        if cur_pos == p2:
                            pos2 = j
                            p2 = -1
                        cur_pos += len(word) + 1
                    for j in range(j + 1, Config.sen_len):
                        cur_ref_data_word[j] = 0
                    if pos1 == -1 or pos2 == -1:
                        raise Exception("[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sentence, head, tail))
                    if pos1 >= Config.sen_len:
                        pos1 = Config.sen_len - 1
                    if pos2 >= Config.sen_len:
                        pos2 = Config.sen_len - 1
                    for j in range(Config.sen_len):
                        data_pos1[i][j] = j - pos1 + Config.sen_len
                        data_pos2[i][j] = j - pos2 + Config.sen_len
                
                np.save(os.path.join("../data/pre_processed_data", "test_word.npy"), data_word)
                np.save(os.path.join("../data/pre_processed_data", "test_pos1.npy"), data_pos1)
                np.save(os.path.join("../data/pre_processed_data", "test_pos2.npy"), data_pos2)
                np.save(os.path.join("../data/pre_processed_data", "test_query.npy"), data_query)
                json.dump(entities_pos_dict, open(os.path.join("../data/pre_processed_data", "test_entpair2scope.json"),'w'))

            print("loading test data")
            self.word_vec = np.load(os.path.join("../data/pre_processed_data", "word_vec.npy"))
            self.data_word = np.load(os.path.join("../data/pre_processed_data", "test_word.npy"))
            self.data_pos1 = np.load(os.path.join("../data/pre_processed_data", "test_pos1.npy"))
            self.data_pos2 = np.load(os.path.join("../data/pre_processed_data", "test_pos2.npy"))
            self.data_query = np.load(os.path.join("../data/pre_processed_data", "test_query.npy"))
            self.entpair2scope = json.load(open(os.path.join("../data/pre_processed_data", "test_entpair2scope.json")))
            Config.rel_num = len(json.load(open(os.path.join("../data/nyt", "rel2id.json"))))
            Config.word_tot = len(self.word_vec) + 2
            Config.word_embeeding_dim = len(self.word_vec[0])

            self.instance_tot = len(self.data_word)
            self.order = list(range(self.instance_tot))
            self.idx = 0

    def train_next_batch(self):
        if self.idx >= len(self.order):
            random.shuffle(self.order)
            self.idx = 0
        idx0 = self.idx
        idx1 = self.idx + Config.batch_size
        if idx1 > len(self.order):
            idx1 = len(self.order)
        self.idx = idx1
        batch_data = {}
        _word = []
        _pos1 = []
        _pos2 = []
        _one_query = []
        _label = []
        _scope = []
        cur_pos = 0
        for i in range(idx0, idx1):
            _word.append(self.train_positive_word[self.positive_scope[self.order[i]][0]:self.positive_scope[self.order[i]][1]])
            _pos1.append(self.train_positive_pos1[self.positive_scope[self.order[i]][0]:self.positive_scope[self.order[i]][1]])
            _pos2.append(self.train_positive_pos2[self.positive_scope[self.order[i]][0]:self.positive_scope[self.order[i]][1]])
            _label.append(self.train_positive_query[self.positive_scope[self.order[i]][0]])
            _one_query.append(self.train_one_query[self.positive_scope[self.order[i]][0]])
            bag_size = self.positive_scope[self.order[i]][1] - self.positive_scope[self.order[i]][0]
            _scope.append([cur_pos, cur_pos + bag_size])
            cur_pos = cur_pos + bag_size
        batch_data['pos_word'] = self.to_tensor(np.concatenate(_word))
        batch_data['pos_pos1'] = self.to_tensor(np.concatenate(_pos1))
        batch_data['pos_pos2'] = self.to_tensor(np.concatenate(_pos2))
        batch_data['pos_label'] = self.to_tensor(np.stack(_label))
        batch_data['pos_query'] = self.to_tensor(np.stack(_one_query))
        batch_data['pos_scope'] = np.stack(_scope)

        # negative sample
        _word = []
        _pos1 = []
        _pos2 = []
        _label = []
        _scope = []
        _multi_label = []
        cur_pos = 0
        for i in range(idx0, idx1):
            _word.append(self.train_negative_word[self.negative_scope[self.order[i]][0]:self.negative_scope[self.order[i]][1]])
            _pos1.append(self.train_negative_pos1[self.negative_scope[self.order[i]][0]:self.negative_scope[self.order[i]][1]])
            _pos2.append(self.train_negative_pos2[self.negative_scope[self.order[i]][0]:self.negative_scope[self.order[i]][1]])
            _multi_label.append(self.train_multi_query[self.negative_scope[self.order[i]][0]:self.negative_scope[self.order[i]][1]])
            _label.append(self.train_negative_query[self.negative_scope[self.order[i]][0]:self.negative_scope[self.order[i]][1]])
            bag_size = self.negative_scope[self.order[i]][1] - self.negative_scope[self.order[i]][0]
            _scope.append([cur_pos, cur_pos + bag_size])
            cur_pos = cur_pos + bag_size
        batch_data['neg_word'] = self.to_tensor(np.concatenate(_word))
        batch_data['neg_pos1'] = self.to_tensor(np.concatenate(_pos1))
        batch_data['neg_pos2'] = self.to_tensor(np.concatenate(_pos2))
        batch_data['mul_label'] = self.to_tensor(np.concatenate(_multi_label))
        batch_data['neg_label'] = self.to_tensor(np.concatenate(_label))
        batch_data['neg_scope'] = np.stack(_scope)

        return batch_data
    
    def test_next_batch(self):
        if self.idx >= len(self.order):
            self.idx = 0
        idx0 = self.idx
        idx1 = self.idx + Config.batch_size
        if idx1 > len(self.order):
            idx1 = len(self.order)
        self.idx = idx1
        index = self.order[idx0:idx1]
        batch_data = {}
        batch_data["word"] = self.to_tensor(self.data_word[index])
        batch_data["pos1"] = self.to_tensor(self.data_pos1[index])
        batch_data["pos2"] = self.to_tensor(self.data_pos2[index])
        return batch_data
    
    def convert(self, data, word2id, rel2id, mode):
        print("begin converting....")
        instance_tot = len(data)
        data_word = np.zeros((instance_tot, Config.sen_len), dtype=int)
        data_pos1 = np.zeros((instance_tot, Config.sen_len), dtype=int)
        data_pos2 = np.zeros((instance_tot, Config.sen_len), dtype=int)
        data_query = np.zeros((instance_tot, Config.rel_num), dtype=int)

        
        for i in range(len(data)):
            instance = data[i]
            head = instance["head"]["word"].lower()
            tail = instance["tail"]["word"].lower()
            sentence = instance["sentence"].lower()
            try:
                data_query[i][rel2id[instance["relation"]]] = 1
            except:
                data_query[i][0] = 1
                print("relation error 1")

            p1 = sentence.find(' ' + head + ' ')
            p2 = sentence.find(' ' + tail + ' ')
            if p1 == -1:
                if sentence[:len(head) + 1] == head + " ":
                    p1 = 0
                elif sentence[-len(head) - 1:] == " " + head:
                    p1 = len(sentence) - len(head)
                else:
                    p1 = 0 # shouldn't happen
            else:
                p1 += 1
            if p2 == -1:
                if sentence[:len(tail) + 1] == tail + " ":
                    p2 = 0
                elif sentence[-len(tail) - 1:] == " " + tail:
                    p2 = len(sentence) - len(tail)
                else:
                    p2 = 0 # shouldn't happen
            else:
                p2 += 1
            words = sentence.split()
            cur_ref_data_word = data_word[i]         
            cur_pos = 0
            pos1 = -1
            pos2 = -1
            for j, word in enumerate(words):
                if j < Config.sen_len:
                    word = word.lower()
                    if word in word2id:
                        cur_ref_data_word[j] = word2id[word]
                    else:
                        cur_ref_data_word[j] = 1
                if cur_pos == p1:
                    pos1 = j
                    p1 = -1
                if cur_pos == p2:
                    pos2 = j
                    p2 = -1
                cur_pos += len(word) + 1
            for j in range(j + 1, Config.sen_len):
                cur_ref_data_word[j] = 0
            if pos1 == -1 or pos2 == -1:
                raise Exception("[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sentence, head, tail))
            if pos1 >= Config.sen_len:
                pos1 = Config.sen_len - 1
            if pos2 >= Config.sen_len:
                pos2 = Config.sen_len - 1
            for j in range(Config.sen_len):
                data_pos1[i][j] = j - pos1 + Config.sen_len
                data_pos2[i][j] = j - pos2 + Config.sen_len
        
  
        np.save(os.path.join("../data/pre_processed_data", "train_"+mode+"_word.npy"), data_word)
        np.save(os.path.join("../data/pre_processed_data", "train_"+mode+"_pos1.npy"), data_pos1)
        np.save(os.path.join("../data/pre_processed_data", "train_"+mode+"_pos2.npy"), data_pos2)
        np.save(os.path.join("../data/pre_processed_data", "train_"+mode+"_query.npy"), data_query)
        
    def to_tensor(self, array):
        return torch.from_numpy(array)

    def genBag(self):
        print("process positive and negative instance....")
        train = json.load(open("../data/nyt/train.json"))
        positive_instance = []
        negative_instance = []
        for ins in train:
            if ins['relation'] == "NA":
                negative_instance.append(ins)
            else:
                positive_instance.append(ins)

        positive_instance.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + "#" + a["relation"])
        negative_instance.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + "#" + a["relation"])

        relfact2scope = {}
        curr_relfact = positive_instance[0]["head"]["id"]+"#"+positive_instance[0]["tail"]["id"]+"#"+positive_instance[0]["relation"]
        relfact2scope[curr_relfact] = [0,]
        for i, instance in enumerate(positive_instance):
            relfact = instance["head"]["id"]+"#"+instance["tail"]["id"] + "#"+instance["relation"]
            if relfact!=curr_relfact:
                relfact2scope[curr_relfact].append(i)
                curr_relfact = relfact
                relfact2scope[curr_relfact] = [i,]
        relfact2scope[curr_relfact].append(len(positive_instance))

        neg_relfact2scope = {}
        curr_relfact = negative_instance[0]["head"]["id"]+"#"+negative_instance[0]["tail"]["id"]+"#"+negative_instance[0]["relation"]
        neg_relfact2scope[curr_relfact] = [0,]
        for i, instance in enumerate(negative_instance):
            relfact = instance["head"]["id"]+"#"+instance["tail"]["id"] + "#"+instance["relation"]
            if relfact!=curr_relfact:
                neg_relfact2scope[curr_relfact].append(i)
                curr_relfact = relfact
                neg_relfact2scope[curr_relfact] = [i,]
        neg_relfact2scope[curr_relfact].append(len(negative_instance))


        entity2scopelist = {}
        for key in neg_relfact2scope.keys():
            ins = negative_instance[neg_relfact2scope[key][0]]
            if entity2scopelist.get(ins['head']['id'], -1) == -1:
                entity2scopelist[ins['head']['id']] = [neg_relfact2scope[key],]
            else:
                entity2scopelist[ins['head']['id']].append(neg_relfact2scope[key])
            if entity2scopelist.get(ins['tail']['id'], -1) == -1:
                entity2scopelist[ins['tail']['id']] = [neg_relfact2scope[key], ]
            else:
                entity2scopelist[ins['tail']['id']].append(neg_relfact2scope[key])
        
        entity2rel = {}
        for ins in positive_instance:
            if ins['head']['id'] not in entity2rel:
                entity2rel[ins['head']['id']] = [ins['relation'],]
            else:
                entity2rel[ins['head']['id']].append(ins['relation'])
            if ins['tail']['id'] not in entity2rel:
                entity2rel[ins['tail']['id']] = [ins['relation'],]
            else:
                entity2rel[ins['tail']['id']].append(ins['relation'])


        '''len(relfact2scope) should equals to len(neg_pos_relfact2scope) '''
        neg_bag_size = 20
        neg_pos_relfact2scope = {}
        neg_pos_relfact2rel = {}
        negative_sample_instance = []
        curr_pos = 0
        relfact2scope_keys = []
        for key in relfact2scope.keys():
            relfact2scope_keys.append(key)
        for i in range(len(relfact2scope_keys)):
            head = positive_instance[relfact2scope[relfact2scope_keys[i]][0]]['head']['id']
            tail = positive_instance[relfact2scope[relfact2scope_keys[i]][0]]['tail']['id']
            head_neg_sample_scopelist = entity2scopelist.get(head, -1)
            tail_neg_sample_scopelist = entity2scopelist.get(tail, -1)
            neg_pos_relfact2rel[relfact2scope_keys[i]] = []
            for rel in entity2rel[head]:
                neg_pos_relfact2rel[relfact2scope_keys[i]].append(rel)
            for rel in entity2rel[tail]:
                if rel not in neg_pos_relfact2rel[relfact2scope_keys[i]]:
                    neg_pos_relfact2rel[relfact2scope_keys[i]].append(rel)
            if head_neg_sample_scopelist == -1 and tail_neg_sample_scopelist == -1:
                flag = i+1 if i+1 < len(relfact2scope_keys) else i-1
                neg_samples = negative_instance[relfact2scope[relfact2scope_keys[flag]][0]:relfact2scope[relfact2scope_keys[flag]][1]]
                negative_sample_instance.extend(neg_samples)
                neg_pos_relfact2scope[relfact2scope_keys[i]] = [curr_pos, curr_pos+len(neg_samples)]
                curr_pos += len(neg_samples)
                continue
            ori_neg_sample_scopelist = []
            if head_neg_sample_scopelist != -1:
                ori_neg_sample_scopelist.extend(head_neg_sample_scopelist)
            if tail_neg_sample_scopelist != -1:
                ori_neg_sample_scopelist.extend(tail_neg_sample_scopelist)
            neg_sample_scopelist = []
            id0 = random.randint(0, len(ori_neg_sample_scopelist)-1)
            step = int(len(ori_neg_sample_scopelist) / neg_bag_size)
            for j in range(neg_bag_size):
                neg_sample_scopelist.append(ori_neg_sample_scopelist[(id0+j*step)%len(ori_neg_sample_scopelist)])

            for scope in neg_sample_scopelist:
                randindex = random.randint(scope[0], scope[1]-1)
                negative_sample_instance.append(negative_instance[randindex])
            neg_pos_relfact2scope[relfact2scope_keys[i]] = [curr_pos, curr_pos+len(neg_sample_scopelist)]
            curr_pos += len(neg_sample_scopelist)

        na_instance = []
        na_relfact2scope = {}

        positive_entities = []
        for ins in positive_instance:
            if ins['head']['id'] not in positive_entities:
                positive_entities.append(ins['head']['id'])
            if ins['tail']['id'] not in positive_entities:
                positive_entities.append(ins['tail']['id'])
        num_ins = 0
        for ins in negative_instance:
            if ins['head']['id'] in positive_entities:
                num_ins += 1
                continue 
            if ins['tail']['id'] in positive_entities:
                num_ins += 1
                continue
            na_instance.append(ins)
        print("the num of negative instances which contain entity in positive instance is: %d" % num_ins)

        na_instance.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + "#" + a["relation"])
        curr_relfact = na_instance[0]["head"]["id"]+"#"+na_instance[0]["tail"]["id"]+"#"+na_instance[0]["relation"]
        na_relfact2scope[curr_relfact] = [0,]
        for i, instance in enumerate(na_instance):
            relfact = instance["head"]["id"]+"#"+instance["tail"]["id"] + "#"+instance["relation"]
            if relfact!=curr_relfact:
                na_relfact2scope[curr_relfact].append(i)
                curr_relfact = relfact
                na_relfact2scope[curr_relfact] = [i,]
        na_relfact2scope[curr_relfact].append(len(na_instance))

        
        positive_instance.extend(na_instance)
        len_relfact = len(relfact2scope)
        for key in na_relfact2scope.keys():
            relfact2scope[key] = [na_relfact2scope[key][0]+len_relfact, na_relfact2scope[key][1]+len_relfact]
        json.dump(positive_instance, open("../data/nyt/postive_train.json", 'w'))
        json.dump(relfact2scope, open("../data/nyt/positive_relfact2scope.json", 'w'))
        print("positive instance saved")
        
        for ins in na_instance:
            ins['relation'] = "/location/location/contains"
        negative_sample_instance.extend(na_instance)
        len_neg_relfact = len(neg_pos_relfact2scope)
        for key in na_relfact2scope.keys():
            neg_pos_relfact2scope[key] = [na_relfact2scope[key][0]+len_neg_relfact, na_relfact2scope[key][1]+len_neg_relfact]
            neg_pos_relfact2rel[key] = ["/location/location/contains"]
        json.dump(negative_sample_instance, open("../data/nyt/negative_train.json", 'w'))
        json.dump(neg_pos_relfact2scope, open("../data/nyt/negative_relfact2scope.json", 'w'))
        json.dump(neg_pos_relfact2rel, open("../data/nyt/negative_relfact2rel.json", 'w'))
        print("negative instance saved")
        

    
            


        

    
            




        
        
