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
from transformers import BertTokenizer
from config import Config
import multiprocessing.dummy as mp 



class Dataloader:
    '''
    # This class 
    '''

    def __init__(self, mode, flag):    
        if not os.path.exists("../data/pre_processed_data"):
            os.mkdir("../data/pre_processed_data")
        if not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_word.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_pos1.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_pos2.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_label.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_length.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_knowledge.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_neg_samples.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_mask.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_entpair2scope.json")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_relfact2scope.json")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_select_mask.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_input_ids.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_attention_mask.npy")):
            print("There dones't exist pre-processed data, pre-processing...")
            start_time = time.time()
            data = json.load(open(os.path.join("../data/nyt", mode+".json")))
            knowledge = json.load(open(os.path.join("../data/knowledge",mode+'.json')))
            ori_word_vec = json.load(open(os.path.join("../data/nyt","word_vec.json")))
            Config.word_tot = len(ori_word_vec) + 2

            # Bert tokenizer
            tokenizer = BertTokenizer.from_pretrained(Config.model_name_or_path, do_lower_case=True)

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
            self.data_word = np.zeros((self.instance_tot, Config.sen_len), dtype=int)
            self.data_pos1 = np.zeros((self.instance_tot, Config.sen_len), dtype=int)
            self.data_pos2 = np.zeros((self.instance_tot, Config.sen_len), dtype=int)
            self.data_length = np.zeros((self.instance_tot, ), dtype=int)
            self.data_label = np.zeros((self.instance_tot, ), dtype=int)
            self.data_knowledge = np.zeros((self.instance_tot, Config.rel_num), dtype=float)
            self.data_mask = np.zeros((self.instance_tot, Config.sen_len), dtype=int)
            self.data_neg_samples = np.zeros((self.instance_tot, Config.neg_samples), dtype=int)
            self.data_select_mask = np.zeros((self.instance_tot, Config.rel_num), dtype=float)
            self.data_input_ids = np.zeros((self.instance_tot, Config.sen_len), dtype=int) - 1
            self.data_attention_mask = np.zeros((self.instance_tot, Config.sen_len), dtype=int)

            def _process_loop(i):
                # for i, instance in enumerate(data):
                instance = data[i]
                sentence = instance["sentence"]
                head = instance["head"]["word"]
                tail = instance["tail"]["word"]
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
                cur_ref_data_word = self.data_word[i]         
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
                self.data_length[i] = len(words)
                if len(words) > Config.sen_len:
                    self.data_length[i] = Config.sen_len
                if pos1 == -1 or pos2 == -1:
                    raise Exception("[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sentence, head, tail))
                if pos1 >= Config.sen_len:
                    pos1 = Config.sen_len - 1
                if pos2 >= Config.sen_len:
                    pos2 = Config.sen_len - 1
                for j in range(Config.sen_len):
                    self.data_pos1[i][j] = j - pos1 + Config.sen_len
                    self.data_pos2[i][j] = j - pos2 + Config.sen_len
                try:
                    self.data_label[i] = rel2id[instance["relation"]]
                except:
                    self.data_label[i] = 0
                
                # input ids for bert
                bert_tokens = tokenizer.tokenize(sentence)
                head_tokens = tokenizer.tokenize(head)
                tail_tokens = tokenizer.tokenize(tail)
                head_pos = bert_tokens.index(head_tokens[0])
                bert_tokens.insert(head_pos, "[unused0]")
                bert_tokens.insert(head_pos+len(head_tokens)+1, "[unused1]")
                tail_pos = bert_tokens.index(tail_tokens[0])
                bert_tokens.insert(tail_pos, "[unused2]")
                bert_tokens.insert(tail_pos+len(tail_tokens)+1, "[unused3]")
                bert_tokens.insert(0, "[CLS]")
                bert_tokens.append("[SEP]")
                length = min(len(bert_tokens), Config.sen_len)
                self.data_input_ids[i][0:length] = tokenizer.convert_tokens_to_ids(bert_tokens[0:length])
                self.data_attention_mask[i][0:length] = 1


                # negative sample for train
                entities = instance["head"]["id"]+"#"+instance["tail"]["id"]
                pos = entities_pos_dict[entities]
                for j in range(Config.neg_samples):
                    index = random.randint(pos[1], self.instance_tot+pos[0]) % self.instance_tot
                    self.data_neg_samples[i][j] = index

                # knowledge 
                rels = knowledge[entities]
                rel_num = len(rels)
                if rel_num != 1 and rels[len(rels)-1]=="NA":
                    rel_num -= 1
                for rel in rels:
                    try:
                        self.data_select_mask[i][rel2id[rel]] = 1                    
                        if len(rels) != 1:
                            if rel2id[rel] == 0:
                                self.data_knowledge[i][rel2id[rel]] = 0
                            else:
                                self.data_knowledge[i][rel2id[rel]] = 1 / rel_num
                        else:
                            self.data_knowledge[i][rel2id[rel]] = 1.0
                    except:
                        continue

                # mask words which are not entities
                head = instance["head"]["word"].split()
                tail = instance["tail"]["word"].split()
                for k in range(pos1, pos1+len(head)):
                    if k < Config.sen_len:
                        self.data_mask[i][k] = 1
                for k in range(pos2, pos2+len(tail)):
                    if k < Config.sen_len:
                        self.data_mask[i][k] = 1
            
            print("begin multiple thread processing...")
            pool = mp.Pool(12)
            pool.map(_process_loop, range(0, self.instance_tot))

            # save array
            np.save(os.path.join("../data/pre_processed_data", mode+"_word.npy"), self.data_word)
            np.save(os.path.join("../data/pre_processed_data", mode+"_pos1.npy"), self.data_pos1)
            np.save(os.path.join("../data/pre_processed_data", mode+"_pos2.npy"), self.data_pos2)
            np.save(os.path.join("../data/pre_processed_data", mode+"_label.npy"), self.data_label)
            np.save(os.path.join("../data/pre_processed_data", mode+"_length.npy"), self.data_length)
            np.save(os.path.join("../data/pre_processed_data", mode+"_neg_samples.npy"), self.data_neg_samples)
            np.save(os.path.join("../data/pre_processed_data", mode+"_knowledge.npy"), self.data_knowledge)
            np.save(os.path.join("../data/pre_processed_data", "word_vec.npy"), self.word_vec)
            np.save(os.path.join("../data/pre_processed_data", mode+"_mask.npy"), self.data_mask)
            np.save(os.path.join("../data/pre_processed_data", mode+"_select_mask.npy"), self.data_select_mask)
            np.save(os.path.join("../data/pre_processed_data", mode+"_input_ids.npy"), self.data_input_ids)
            np.save(os.path.join("../data/pre_processed_data", mode+"_attention_mask.npy"), self.data_attention_mask)
            json.dump(self.entpair2scope, open(os.path.join("../data/pre_processed_data", mode+"_entpair2scope.json"), 'w'))
            json.dump(self.relfact2scope, open(os.path.join("../data/pre_processed_data", mode+"_relfact2scope.json"), "w"))
            print("end pre-process")
            end_time = time.time()
            print(end_time-start_time)
        else:
            print("There exists pre-processed data already. loading....")
            self.data_word = np.load(os.path.join("../data/pre_processed_data", mode+"_word.npy"))
            self.data_pos1 = np.load(os.path.join("../data/pre_processed_data", mode+"_pos1.npy"))
            self.data_pos2 = np.load(os.path.join("../data/pre_processed_data", mode+"_pos2.npy"))
            self.data_label = np.load(os.path.join("../data/pre_processed_data", mode+"_label.npy"))
            self.data_length = np.load(os.path.join("../data/pre_processed_data", mode+"_length.npy"))
            self.data_neg_samples = np.load(os.path.join("../data/pre_processed_data", mode+"_neg_samples.npy"))
            self.data_knowledge = np.load(os.path.join("../data/pre_processed_data", mode+"_knowledge.npy"))
            self.word_vec = np.load(os.path.join("../data/pre_processed_data", "word_vec.npy"))
            self.data_mask = np.load(os.path.join("../data/pre_processed_data", mode+"_mask.npy"))
            self.data_select_mask = np.load(os.path.join("../data/pre_processed_data", mode+"_select_mask.npy"))
            self.data_input_ids = np.load(os.path.join("../data/pre_processed_data", mode+"_input_ids.npy"))
            self.data_attention_mask = np.load(os.path.join("../data/pre_processed_data", mode+"_attention_mask.npy"))
            self.entpair2scope = json.load(open(os.path.join("../data/pre_processed_data", mode+"_entpair2scope.json")))
            self.relfact2scope = json.load(open(os.path.join("../data/pre_processed_data", mode+"_relfact2scope.json")))
            Config.word_tot = self.word_vec.shape[0] + 2
            Config.rel_num = len(json.load(open(os.path.join("../data/nyt", "rel2id.json"))))
            Config.word_embeeding_dim = self.word_vec.shape[1]
            print("Finish loading...")
            self.instance_tot = self.data_word.shape[0]
        self.entpair_tot = len(self.entpair2scope)
        self.relfact_tot = len(self.relfact2scope)
        
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
        # random.shuffle(self.order)

        # weight for train crossEntropyloss
        self.weight = np.zeros((Config.rel_num), dtype=float)
        for i in self.data_label:
            self.weight[i] += 1
        self.weight = 1 / self.weight**0.05

    def next_batch(self):
        if self.idx >= len(self.order):
            # if training
            if Config.training:
                random.shuffle(self.order)
            self.idx = 0
        batch_data = {} 
        idx0 = self.idx
        idx1 = self.idx + Config.batch_size
        if idx1 > len(self.order):
            idx1 = len(self.order)
        self.idx = idx1
        if self.flag == "ins":
            index = self.order[idx0:idx1]
            batch_data["pos_word"] = self.data_word[index]
            batch_data["pos_pos1"] = self.data_pos1[index]
            batch_data["pos_pos2"] = self.data_pos2[index]
            batch_data["query"] = self.data_label[index]
            batch_data["mask"] = self.data_mask[index]
            batch_data["knowledge"] = self.data_knowledge[index]
            batch_data["select_mask"] = self.data_select_mask[index]
            batch_data["input_ids"] = self.data_input_ids[index]
            batch_data["attention_mask"] = self.data_attention_mask[index]
            # neg sample (batch_size, neg_samples, sen_len)
            neg_indexes = self.data_neg_samples[index]
            neg_samples = Config.neg_samples
            if neg_samples > neg_indexes.shape[1]:
                neg_samples = neg_indexes.shape[1]
            neg_indexes = neg_indexes[:, :neg_samples]
            batch_data["neg_word"] = self.data_word[neg_indexes]
            batch_data["neg_pos1"] = self.data_pos1[neg_indexes]
            batch_data["neg_pos2"] = self.data_pos2[neg_indexes]
            batch_data["scope"] = None
            multi_rel = np.zeros((idx1-idx0, Config.rel_num), dtype=np.int32)
            for i in range(idx1-idx0):
                multi_rel[i][self.data_label[index[i]]] = 1
            batch_data["multi_label"] = multi_rel
            return batch_data
        else:
            _word = []
            _pos1 = []
            _pos2 = []
            _ids = []
            _mask = []
            _rel = []
            _multi_rel = []
            _scope = []
            cur_pos = 0
            for i in range(idx0, idx1):
                _word.append(self.data_word[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _pos1.append(self.data_pos1[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _pos2.append(self.data_pos2[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _ids.append(self.data_input_ids[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _mask.append(self.data_attention_mask[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _rel.append(self.data_label[self.scope[self.order[i]][0]])
                bag_size = self.scope[self.order[i]][1] - self.scope[self.order[i]][0]
                _scope.append([cur_pos, cur_pos + bag_size])
                cur_pos = cur_pos + bag_size
                _one_multi_rel = np.zeros((Config.rel_num), dtype=np.int32)
                for j in range(self.scope[self.order[i]][0], self.scope[self.order[i]][1]):
                    _one_multi_rel[self.data_label[j]] = 1
                _multi_rel.append(_one_multi_rel)
            
            batch_data['pos_word'] = np.concatenate(_word)
            batch_data['pos_pos1'] = np.concatenate(_pos1)
            batch_data['pos_pos2'] = np.concatenate(_pos2)
            batch_data['input_ids'] = np.concatenate(_ids)
            batch_data['attention_mask'] = np.concatenate(_mask)
            batch_data['query'] = np.stack(_rel)
            batch_data['multi_label'] = np.stack(_multi_rel)
            batch_data['scope'] = np.stack(_scope)

            return batch_data
            
            # if Config.training and Config.down_size:
            #     # Down-sizing 
            #     rel = (np.stack(_rel)).tolist()
            #     ins_sample_index = []
            #     bag_sample_index = []
            #     non_na_num = 1
            #     na_num = 0
            #     for i, r in enumerate(rel):
            #         if r != 0:
            #             bag_sample_index.append(i)
            #             non_na_num += 1
            #     for i, r in enumerate(rel):
            #         if r == 0:
            #             bag_sample_index.append(i)
            #             na_num += 1
            #         if na_num >= 3 * non_na_num:
            #             break
            #     scope = []
            #     cur_pos = 0
            #     for i, index in enumerate(bag_sample_index):
            #         # process scope
            #         bag_size = _scope[index][1] - _scope[index][0]
            #         scope.append([cur_pos, cur_pos+bag_size])
            #         cur_pos += bag_size
            #         # process ins-sample
            #         ins_sample_index.extend(list(range(_scope[index][0], _scope[index][1])))

            #     batch_data['pos_word'] = np.concatenate(_word)[ins_sample_index]
            #     batch_data['pos_pos1'] = np.concatenate(_pos1)[ins_sample_index]
            #     batch_data['pos_pos2'] = np.concatenate(_pos2)[ins_sample_index]
            #     batch_data['mask'] = np.concatenate(_mask)[ins_sample_index]
            #     batch_data["select_mask"] = np.concatenate(_select_mask)[ins_sample_index]
            #     batch_data['length'] = np.concatenate(_length)[ins_sample_index]
            #     batch_data["knowledge"] = np.concatenate(_knowledge)[ins_sample_index]
            #     batch_data['label'] = np.stack(_rel)[bag_sample_index]
            #     batch_data["ins_label"] = np.concatenate(_ins_rel)[ins_sample_index]
            #     batch_data['multi_label'] = np.stack(_multi_rel)[bag_sample_index]
            #     batch_data['scope'] = np.stack(scope)
            #     batch_data['bag_knowledge'] = np.stack(_bag_knowledge)[bag_sample_index]
            #     # neg_samples.size(): (batch_size, neg_samples, sen_len)
            #     neg_indexes = np.concatenate(_neg_index)[ins_sample_index]
            #     neg_samples = Config.neg_samples
            #     if neg_samples > neg_indexes.shape[1]:
            #         neg_samples = neg_indexes.shape[1]
            #     neg_indexes = neg_indexes[:, :neg_samples]
            #     batch_data["neg_word"] = self.data_word[neg_indexes]
            #     batch_data["neg_pos1"] = self.data_pos1[neg_indexes]
            #     batch_data["neg_pos2"] = self.data_pos2[neg_indexes]
            # else:


        

    
            




        
        