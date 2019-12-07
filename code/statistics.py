#### analyze the dataset's statistics
import json 
import os 
import math 
import matplotlib
# Use 'Agg' so this program could run on a remote server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb 


def draw(result, filename):
    data = []
    for res in result:
        if math.isnan(res[1]):
            res[1] = 0
            data.append(res)
        else:
            data.append(res)
    data.sort(key=lambda a: a[1], reverse=True)
    x = []
    y = []
    for item in data:
        x.append(item[0])
        y.append(item[1])
    for rect in plt.bar(range(len(y)), y, color='b', tick_label=x):
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2, 1.0*height, "")
    plt.xticks(rotation=90)
    plt.tick_params(labelsize=0)
    plt.savefig(filename)
    plt.close()

def _save(result, filename):
    result.sort(key=lambda a: a[1], reverse=True)
    json.dump(result, open(filename, 'w'))


def _anaylse(dataset, mode):
    data = json.load(open(os.path.join(dataset, mode)))
    # sort data and get scope
    print("sorting data...")
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

    # NA prop
    print("begin analyse NA prop...")
    na_ins_tot = 0
    na_bag_tot = 0
    not_na_ins_tot = 0
    not_na_bag_tot = 0
    for ins in data:
        if ins["relation"] == "NA" or ins["relation"] == "P0":
            na_ins_tot += 1
        else:
            not_na_ins_tot += 1
    for key in relfact2scope.keys():
        scope = relfact2scope[key]
        ins = data[scope[0]]
        if ins["relation"] == "NA" or ins["relation"] == "P0":
            na_bag_tot += 1
        else:
            not_na_bag_tot += 1
    print("NA bag prop: %f" % (na_bag_tot / (na_bag_tot + not_na_bag_tot)))
    print("NA ins prop: %f" % (na_ins_tot / (na_ins_tot + not_na_ins_tot)))
    
    # rel distribution
    print("begin analyse rel distribution...")
    bag_rel2num = {}
    ins_rel2num = {}
    rel2id = json.load(open(os.path.join(dataset, "rel2id.json")))
    for rel in rel2id:
        bag_rel2num[rel] = 0
        ins_rel2num[rel] = 0
    for ins in data:
        rel = ins["relation"]
        if rel in rel2id.keys():
            ins_rel2num[rel] += 1
    for key in relfact2scope.keys():
        scope = relfact2scope[key]
        rel = data[scope[0]]["relation"]
        if rel in rel2id.keys():
            bag_rel2num[rel] += 1
    bag_rel_result = []
    ins_rel_result = []
    for key in bag_rel2num.keys():
        if key == 'NA' or key == "P0":
            continue
        bag_rel_result.append([rel2id[key], bag_rel2num[key]])
        ins_rel_result.append([rel2id[key], ins_rel2num[key]])
    _save(bag_rel_result, dataset+"_"+mode.split(".")[0]+"_bag_rel_dist.json")
    _save(ins_rel_result, dataset+"_"+mode.split(".")[0]+"_ins_rel_dist.json")
    # draw(bag_rel_result, dataset+"_bag_rel_dist.pdf")
    # draw(ins_rel_result, dataset+"_ins_rel_dist.pdf")

    # NA anaylse
    print("begin analyse entity sharing...")
    ins_share_ent_num = 0
    bag_share_ent_num = 0
    na_entity = []
    bag_entity = [] # just for test
    for ins in data:
        if ins["relation"] != "NA" and ins["relation"] != "P0":
            continue
        if ins["head"]["id"] not in na_entity:
            na_entity.append(ins["head"]["id"])
        if ins["tail"]["id"] not in na_entity:
            na_entity.append(ins["tail"]['id'])
    for key in relfact2scope.keys():
        scope = relfact2scope[key]
        ins = data[scope[0]]
        if ins["relation"] != "NA" and ins["relation"] != "P0":
            continue
        if ins["head"]["id"] not in bag_entity:
            bag_entity.append(ins["head"]["id"])
        if ins["tail"]["id"] not in bag_entity:
            bag_entity.append(ins["tail"]['id'])
    if bag_entity != na_entity:
        exit("error")
    for ins in data:
        if ins["relation"] == "NA" or ins["relation"] == "P0":
            continue
        if ins["relation"] not in rel2id.keys():
            continue
        if ins["head"]["id"] in na_entity or ins["tail"]["id"] in na_entity:
            ins_share_ent_num += 1
    for rel in relfact2scope.keys():
        scope = relfact2scope[rel]
        ins = data[scope[0]]
        if ins["relation"] == "NA" or ins["relation"] == "P0":
            continue
        if ins["relation"] not in rel2id.keys():
            continue
        if ins["head"]["id"] in na_entity or ins["tail"]["id"] in na_entity:
            bag_share_ent_num += 1
    print("positive ins shared entity: %f" %(ins_share_ent_num/not_na_ins_tot))
    print("positive bag shared entity: %f" %(bag_share_ent_num/not_na_bag_tot))

    
    # bag size anaylse 

    
def anaylse(dataset):
    if dataset == "wiki":
        _anaylse(dataset, "test.json")
        _anaylse(dataset, "dev.json")
        _anaylse(dataset, "train.json")
    elif dataset == "nyt":
        _anaylse(dataset, "test.json")
        _anaylse(dataset, "train.json")



def sort(data):
    print("sorting data...")
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
    return entities_pos_dict, relfact2scope


def check():
    train = json.load(open(os.path.join("wiki", "train.json")))
    train_entpair2scope, train_relfact2scope = sort(train)
    dev = json.load(open(os.path.join("wiki", "dev.json")))
    dev_entpair2scope, dev_relfact2scope = sort(dev)
    test = json.load(open(os.path.join("wiki", "test.json")))
    test_entpair2scope, test_relfact2scope = sort(test)
    dev_wrong = []
    test_wrong = []
    for key in train_relfact2scope.keys():
        if dev_relfact2scope.get(key, -1) != -1:
            dev_wrong.append(key)
        if test_relfact2scope.get(key, -1) != -1:
            test_wrong.append(key)
    pdb.set_trace()

if __name__ == "__main__":
    anaylse("wiki")
    # check()
