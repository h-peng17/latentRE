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
            head["word"] = ori_data[rel][i]["triple"]["h"]["name"]
            head["id"] = ori_data[rel][i]["triple"]["h"]["id"]
            tail["word"] = ori_data[rel][i]["triple"]["t"]["name"]
            tail["id"] = ori_data[rel][i]["triple"]['t']['name']
            ins["sentence"] = sen
            ins["head"] = head
            ins["tail"] = tail
            ins["relation"] = rel
            data.append(ins)
    return data 
    # json.dump(data, open(os.path.join("../data/wiki", mode), 'w'))

def wash(ori_data):
    data = []
    for ins in ori_data:
        if ins["relation"] == "P0":
            continue
        data.append(ins)
    return data

def sample_na(mode, NA_data, NA_relfact2scope):
    data = json.load(open(os.path.join("../data/wiki", mode)))
    data = wash(data)
    ori_length = len(data)
    data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + "#" + a["relation"])   
    relfact2scope = {}
    curr_relfact = data[0]["head"]["id"]+"#"+data[0]["tail"]["id"]+"#"+data[0]["relation"]
    relfact2scope[curr_relfact] = [0,]
    for i, instance in enumerate(data):
        relfact = instance["head"]["id"]+"#"+instance["tail"]["id"] + "#"+instance["relation"]
        if relfact!=curr_relfact:
            relfact2scope[curr_relfact].append(i)
            curr_relfact = relfact
            relfact2scope[curr_relfact] = [i,]
    relfact2scope[curr_relfact].append(len(data))
    
    not_na_ent = []
    for key in relfact2scope.keys():
        scope = relfact2scope[key]
        ins = data[scope[0]]
        if ins["relation"] != "P0":
            not_na_ent.append(ins["head"]['id'])
            not_na_ent.append(ins["tail"]['id'])
    
    share_scope = []
    for key in NA_relfact2scope.keys():
        scope = NA_relfact2scope[key]
        if scope == -1:
            continue
        ins = NA_data[scope[0]]
        if ins["head"]["id"] in not_na_ent:
            not_na_ent.remove(ins["head"]["id"])
            share_scope.append(scope)
            NA_relfact2scope[key] = -1
        elif ins["tail"]["id"] in not_na_ent:
            not_na_ent.remove(ins["tail"]["id"])
            share_scope.append(scope)
            NA_relfact2scope[key] = -1

    for scope in share_scope:
        data.extend(NA_data[scope[0]:scope[1]])

    print("generate %d instance" % (len(data)-ori_length))
    np.dump(data, open(os.path.join("../data/wiki", mode), 'w'))
    return NA_relfact2scope

def format_sample():
    data = convert("NA.json", "")
    data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + "#" + a["relation"])
    relfact2scope = {}
    curr_relfact = data[0]["head"]["id"]+"#"+data[0]["tail"]["id"]+"#"+data[0]["relation"]
    relfact2scope[curr_relfact] = [0,]
    for i, instance in enumerate(data):
        relfact = instance["head"]["id"]+"#"+instance["tail"]["id"] + "#"+instance["relation"]
        if relfact!=curr_relfact:
            relfact2scope[curr_relfact].append(i)
            curr_relfact = relfact
            relfact2scope[curr_relfact] = [i,]
    relfact2scope[curr_relfact].append(len(data))

    relfact2scope =  sample_na("train.json", data, relfact2scope)
    relfact2scope =  sample_na("dev.json", data, relfact2scope)
    relfact2scope =  sample_na("test.json", data, relfact2scope)

def random_sample():
    train = wash(json.load(open("../data/wiki/train.json")))
    dev = wash(json.load(open("../data/wiki/dev.json")))
    test = wash(json.load(open("../data/wiki/test.json")))
    data = convert("NA.json", "")
    # sort
    print("sorting....")
    data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + "#" + a["relation"])
    relfact2scope = {}
    curr_relfact = data[0]["head"]["id"]+"#"+data[0]["tail"]["id"]+"#"+data[0]["relation"]
    relfact2scope[curr_relfact] = [0,]
    for i, instance in enumerate(data):
        relfact = instance["head"]["id"]+"#"+instance["tail"]["id"] + "#"+instance["relation"]
        if relfact!=curr_relfact:
            relfact2scope[curr_relfact].append(i)
            curr_relfact = relfact
            relfact2scope[curr_relfact] = [i,]
    relfact2scope[curr_relfact].append(len(data))

    print("begin sample...")
    train_sample_num = 500000
    dev_sample_num = int(train_sample_num * (len(dev)/len(train)))
    test_sample_num = int(train_sample_num * (len(test)/len(train)))
    train_scope = []
    dev_scope = []
    test_scope = []
    num_train = 0
    num_dev = 0
    num_test = 0
    for key in relfact2scope.keys():
        scope = relfact2scope[key]
        if num_train < train_sample_num:
            train_scope.append(scope)
            num_train += scope[1] - scope[0]
        elif num_dev < dev_sample_num:
            dev_scope.append(scope)
            num_dev += scope[1] - scope[0]
        elif num_test < test_sample_num:
            test_scope.append(scope)
            num_test += scope[1] - scope[0]
    for scope in train_scope:
        train.extend(data[scope[0]:scope[1]])
    for scope in dev_scope:
        dev.extend(data[scope[0]:scope[1]])
    for scope in test_scope:
        test.extend(data[scope[0]:scope[1]])

    json.dump(train, open("../data/wiki/train.json", "w"))
    json.dump(dev, open("../data/wiki/dev.json", "w"))
    json.dump(test, open("../data/wiki/test.json", "w"))


random_sample()





    

    
    



