
import os
import numpy as np 
import json 
import sklearn.metrics
import pdb 
import math 
import matplotlib
# Use 'Agg' so this program could run on a remote server
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        log(auc, f1, relid, precision)
        return auc

def log(auc, f1, relid, precision):
    if not os.path.exists("../res"):
        os.mkdir("../res")
    f = open(os.path.join("../res", "bert+nyt_log"), 'a+')
    f.write("relation id: %d\n" % relid)
    f.write("auc = "+str(auc)+"| "+"F1 = \n"+str(f1))
    f.write('P@100: {} | P@200: {} | P@300: {} | Mean: {}\n'.format(precision[100], precision[200], precision[300], (precision[100] + precision[200] + precision[300]) / 3))
    f.write("---------------------------------------------------------")
    f.close()

def draw(result, filename):
    print("processing " + filename)
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
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, "")
    plt.xticks(rotation=90)
    plt.tick_params(labelsize=0)
    plt.savefig(os.path.join("../pic", filename+".pdf"))
    plt.close()

def _save(result, filename):
    data = []
    for res in result:
        if math.isnan(res[1]):
            res[1] = 0
            data.append(res)
        else:
            data.append(res)
    data.sort(key=lambda a: a[1], reverse=True)
    json.dump(data, open(filename, 'w'))


def drawAll(logit_name):
    rel2id = json.load(open("../data/wiki/rel2id.json"))
    logit = np.load(os.path.join("../res", logit_name))
    label = np.load("../res/label.npy")
    result = []
    for i in range(1, len(rel2id)):
        _result = []
        _result.append(i)
        auc = eval(logit, label, i)
        _result.append(auc)
        result.append(_result)
    _save(result, "wiki_bert_result.json")
    # draw(result, logit_name.split(".")[0])


def main():
    logit_list = ["bert_logit.npy"]
    for logit_name in logit_list:
        drawAll(logit_name)
    
main()


def test_eval(logit, label, mode):
        res_list = []
        tot = 0
        for i in range(len(logit)):
            for j in range(1, len(logit[i])):
                tot += label[i][j]
                res_list.append([logit[i][j], label[i][j]])
                
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
        print("auc = "+str(auc)+"| "+"F1 = "+str(f1))
        print('P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(precision[100], precision[200], precision[300], (precision[100] + precision[200] + precision[300]) / 3))
        
        plt.plot(recall, precision, lw=2, label=mode)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.3, 1.0])
        plt.xlim([0.0, 0.4])
        plt.title('Precision-Recall')
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(os.path.join("../pic", 'nyt_pr_curve.png'))
        # plt.close()

# def main():
#     logits = ["bert_logit.npy", "cnn+att_logit.npy", "pcnn+att_logit.npy"]
#     label = np.load("../res/label.npy")
#     for file in logits:
#         logit = np.load(os.path.join("../res", file))
#         test_eval(logit, label, file.split(".")[0])

# main()