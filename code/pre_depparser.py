
import os 
import json 
import pdb
import stanfordnlp
import numpy as np 
from tqdm import trange
from config import Config
import sys 
import time
import argparse 

def depparser(index, iter, data):
    nlp = stanfordnlp.Pipeline()
    parser_tokened_word = []
    parser_governor = []
    for i in iter:
        instance = data[i]
        sen = instance["sentence"].lower().replace(".", '')
        tokened_word = []
        governor = []
        doc = nlp(sen)
        words = doc.sentences[0].words
        for word in words:
            tokened_word.append(word.text)
            governor.append(word.governor)
        parser_tokened_word.append(tokened_word)
        parser_governor.append(governor)
        sys.stdout.write("processed: %.3f\r" % (i / 255000))
        sys.stdout.flush()
    json.dump(parser_tokened_word, open("../data/parser_tokened_word"+str(index)+".json", 'w'))
    json.dump(parser_governor, open("../data/parser_governor"+str(index)+".json", 'w'))

parser = argparse.ArgumentParser(description="latentRE")
parser.add_argument("--cuda", dest="cuda", type=str, default="4", help="cuda")
parser.add_argument("--index", dest="index", type=int, default=0, help="index")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
data = json.load(open("../data/nyt/train.json"))
data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + "#" + a["relation"])
if args.index == 0:
    iter = range(0, 255000)
    depparser(0, iter, data)
elif args.index == 1:
    iter = range(255000, len(data))
    depparser(1, iter, data)
    


        

        
