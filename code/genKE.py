"""
This file is to gen knowledge
"""
import os 
import json 


if not os.path.exists("../data/knowledge"):
    os.mkdir("../data/knowledge")


def gen_knowledge(mode):
    rel2id = json.load(open("../data/nyt/rel2id.json"))
    data = json.load(open(os.path.join("../data/nyt", mode+".json")))
    print('processing...')
    knowledge = {}
    for i in range(len(data)):
        instance = data[i]
        rel = instance["relation"]
        entities = instance["head"]["id"]+"#"+instance["tail"]["id"]
        if entities not in knowledge:
            knowledge[entities] = []
            knowledge[entities].append(rel)
        else:
            if rel not in knowledge[entities]:
                knowledge[entities].append(rel)
    
    print('sav...')
    json.dump(knowledge, open(os.path.join("../data/knowledge", mode+".json"), 'w'))


gen_knowledge("train")
gen_knowledge("test")