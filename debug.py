import os 
import json
import pdb


def debug():
    mask = json.load(open("depparsermask.json"))
    input = json.load(open("depparserinput.json"))
    mask_word = []
    input_word =[]
    for i in range(len(mask)):
        _mask = []
        _input_word = []
        for item in mask[i]:
            if item != 0:
                _mask.append(input[i].index(item))
            else:
                _mask.append(-1)
        for item in input[i]:
            if item == '[unused0]' or item == '[unused1]' or item == '[unused2]' or item == '[unused3]':
                continue
            else:
                _input_word.append(item)
        for j, item in enumerate(_mask):
            if item != -1:
                try:
                    _mask[j] = _input_word[item]
                except:
                    print("warning")
                    continue
        mask_word.append(_mask)
        input_word.append(_input_word)
    json.dump(mask_word, open("depparsermask1.json", 'w'))
    json.dump(input_word, open("depparserinput1.json", 'w'))

debug()
                