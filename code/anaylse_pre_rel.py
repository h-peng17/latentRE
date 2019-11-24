
import os 
import numpy as np 
import json 
import sklearn.metrics


def eval(logit, label, relid):
        res_list = []
        tot = 0
        for i in range(len(logit)):
            tot += label[i][relid]
            res_list.append([logit[i][relid], label[i][relid]])
                
        #sort res_list
        res_list.sort(key=lambda val: val[0], reverse=True)
        precision = np.zeros((len(res_list)), dtype=float)
        recall = np.zeros((len(res_list)), dtype=float)
        corr = 0
        for i, res in enumerate(res_list):
            corr += res[1]
            precision[i] = corr/(i+1)
            recall[i] = corr/tot
        
        f1 = (2*precision*recall / (recall+precision+1e-20)).max()
        auc = sklearn.metrics.auc(x=recall, y=precision)
        # print("---------------------------------------------------------")
        # print("relation id: %d" % relid)
        # print("auc = "+str(auc)+"| "+"F1 = "+str(f1))
        # print('P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(precision[100], precision[200], precision[300], (precision[100] + precision[200] + precision[300]) / 3))
        # print("---------------------------------------------------------")
        log(auc, f1, relid, precision)
        return auc

def log(auc, f1, relid, precision):
    if not os.path.exists("../res"):
        os.mkdir("../res")
    f = open(os.path.join("../res", "bert+nyt_log"), 'a+')
    f.write("---------------------------------------------------------")
    f.write("relation id: %d\n" % relid)
    f.write("auc = "+str(auc)+"| "+"F1 = \n"+str(f1))
    f.write('P@100: {} | P@200: {} | P@300: {} | Mean: {}\n'.format(precision[100], precision[200], precision[300], (precision[100] + precision[200] + precision[300]) / 3))
    f.write("---------------------------------------------------------")
    f.close()

rel2id = json.load(open("../data/nyt/rel2id.json"))
logit = np.load("../res/bert+nyt_logit.npy")
label = np.load("../res/pcnn+att+wiki_label.npy")
result = []
for i in range(1, len(rel2id)):
    _result = []
    _result.append(i)
    auc = eval(logit, label, i)
    _result.append(auc)
    result.append(_result)
json.dump(result, open(os.path.join("../res", "bert+nyt_result.json"), 'w'))
