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
import multiprocessing.dummy as mp 


class Dataloader:
    '''
    # This class 
    '''

    def __init__(self, mode):    
        if not os.path.exists("../data/pre_processed_data"):
            os.mkdir("../data/pre_processed_data")
        if not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_word.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_pos1.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_pos2.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_label.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_length.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_knowledge.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_neg_samples.npy")) or \
        not os.path.exists(os.path.join("../data/pre_processed_data", mode+"_mask.npy")):
            print("There dones't exist pre-processed data, pre-processing...")
            start_time = time.time()
            data = json.load(open(os.path.join("../data/nyt", mode+".json")))
            knowledge = json.load(open(os.path.join("../data/knowledge",mode+'.json')))
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
            self.word_vec = np.asarray(word_vec)
            Config.word_embeeding_dim = len(word_vec[0])
            
            # sort data by head and tail and get entities-pos dict          
            data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'])   
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
                # negative sample for train
                entities = instance["head"]["id"]+"#"+instance["tail"]["id"]
                pos = entities_pos_dict[entities]
                for j in range(Config.neg_samples):
                    index = random.randint(pos[1], self.instance_tot+pos[0]) % self.instance_tot
                    self.data_neg_samples[i][j] = index

                # knowledge 
                rels = knowledge[entities]
                for rel in rels:
                    try:
                        self.data_knowledge[i][rel2id[rel]] = 1.0 / len(rels)
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
            Config.word_tot = self.word_vec.shape[0] + 2
            Config.rel_num = len(json.load(open(os.path.join("../data/nyt", "rel2id.json"))))
            Config.word_embeeding_dim = self.word_vec.shape[1]
            print("Finish loading...")
            self.instance_tot = self.data_word.shape[0]
        self.order = list(range(self.instance_tot))
        self.idx = 0
        random.shuffle(self.order)
        self.weight = np.ones((Config.rel_num), dtype=float)
        for i in self.data_label:
            self.weight[i] += 1
        for i in range(Config.rel_num):
            self.weight[i] = 1 / (self.weight[i]**0.5)


    def next_batch(self):
        if self.idx >= len(self.order):
            random.shuffle(self.order)
            self.idx = 0

        batch_data = {} 
        idx0 = self.idx
        idx1 = self.idx + Config.batch_size
        if idx1 > len(self.order):
            idx1 = len(self.order)
        self.idx = idx1
        index = self.order[idx0:idx1]
        batch_data["pos_word"] = self.data_word[index]
        batch_data["pos_pos1"] = self.data_pos1[index]
        batch_data["pos_pos2"] = self.data_pos2[index]
        batch_data["label"] = self.data_label[index]
        batch_data["length"] = self.data_length[index]
        batch_data["knowledge"] = self.data_knowledge[index]
        # neg_samples.size(): (batch_size, neg_samples, sen_len)
        neg_indexes = self.data_neg_samples[index]
        batch_data["neg_word"] = self.data_word[neg_indexes]
        batch_data["neg_pos1"] = self.data_pos1[neg_indexes]
        batch_data["neg_pos2"] = self.data_pos2[neg_indexes]
        batch_data["mask"] = self.data_mask[index]

        return batch_data

        

    
            




        
        