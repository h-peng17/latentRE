# This file is to convert wiki format data to nyt format data
import json 
import os 


def convert(filename, mode):
    ori_data = json.load(open(os.path.join("../data/wiki", filename)))
    data = []
    for rel in ori_data.keys():
        for i in range(len(ori_data[rel])):
            ins = {}
            sen = " ".join(ori_data[rel][i]["tokens"])
            head = {}
            tail = {}
            head["word"] = ori_data[rel][i]["h"]["name"]
            head["id"] = ori_data[rel][i]["h"]["id"]
            tail["word"] = ori_data[rel][i]["t"]["name"]
            tail["id"] = ori_data[rel][i]['t']['name']
            ins["sentence"] = sen
            ins["head"] = head
            ins["tail"] = tail
            ins["relation"] = rel
            data.append(ins)
    json.dump(data, open(os.path.join("../data/wiki", mode), 'w'))

# convert("train_set.json", "train.json")
# convert("val_set.json","dev.json")
# convert("common_test_set.json", "test1.json")
# convert("uncommon_test_set.json", "test2.json")

data = json.load(open("../data/wiki/train_set.json"))
rel2id = {}
for rel in data.keys():
    rel2id[rel] = len(rel2id)

json.dump(rel2id, open("../data/wiki/rel2id.json", "w"))