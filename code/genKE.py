"""
This file is to gen knowledge
"""
import os 
import json 


if not os.path.exists("../data/knowledge"):
    os.mkdir("../data/knowledge")


def gen_knowledge(mode, dataset):
    rel2id = json.load(open("../data/"+dataset+"/rel2id.json"))
    data = json.load(open(os.path.join("../data/"+dataset, mode+".json")))
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
    json.dump(knowledge, open(os.path.join("../data/knowledge", dataset+"_"+mode+".json"), 'w'))

dataset = argv[1]
gen_knowledge("train", dataset)
gen_knowledge("test", dataset)