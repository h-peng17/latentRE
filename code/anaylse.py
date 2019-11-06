from transformers import GPT2Tokenizer
import json 
import os 


def convert(filelist):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    ori_input_words = json.load(open(os.path.join("../output", filelist[0])))
    ori_mask = json.load(open(os.path.join("../output", filelist[1])))
    ori_pre = json.load(open(os.path.join("../output", filelist[2])))
    input_words = []
    mask = []
    pre = []
    for batch in ori_input_words:
        for ins in batch:
            input_words.append(tokenizer.decode(ins))
    for batch in ori_pre:
        for ins in batch:
            pre.append(tokenizer.decode(ins))
    json.dump(input_words, open(os.path.join("../output", filelist[0].split(".")[0]+"_after.json"), 'w'))
    json.dump(pre, open(os.path.join("../output", filelist[2].split(".")[0]+"_after.json"), 'w'))

filelist = ["01gengpt2input_words.npy", "01gengpt2mask.npy", "01gengpt2pre_words.npy"]
convert(filelist)