import pandas as pd
import numpy as np
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

dfdev= pd.read_csv('te.translit.sampled.dev.tsv',sep='\t')
dftest = pd.read_csv('te.translit.sampled.test.tsv',sep='\t')
dft = pd.read_csv('te.translit.sampled.train.tsv',sep='\t')
dft.columns = ['tel','eng','rep']
dfdev.columns = ['tel','eng','rep']
dftest.columns = ['tel','eng','rep']
dft = dft.dropna()


def build_vocabulary(words):
    dic = {}
    k = 0
    for i in range(len(words)):
        word = words[i].lower()
        for char in word:
            if char not in dic:
                dic[char]= k
                k+=1
    return dic
eng = list(dft['eng'])
tel = list(dft['tel'])
tel1 = list(dfdev['tel'])
tel2 = list(dftest['tel'])
tel = tel+tel1+tel2
eng_voc = build_vocabulary(eng)
tel_voc = build_vocabulary(tel)


addlists = ["<eos>", "<pad>", "<go>", "<unk>"]
j = len(eng_voc)
for i in addlists:
    eng_voc[i]=j
    j+=1
k =len(tel_voc)
for i in addlists:
    tel_voc[i]=k
    k+=1

def get_tokens(column, tar, voc):
    train = []
    for word in column:
        indices = [voc.get(char, voc["<unk>"]) for char in word.lower()]
        if tar:
            indices.append(voc["<eos>"])
        train.append(indices)
    return train
xtrain = get_tokens(dft["eng"], 0, eng_voc)
ytrain = get_tokens(dft['tel'],1,tel_voc)
xval = get_tokens(dfdev["eng"], 0, eng_voc)
yval = get_tokens(dfdev['tel'],1,tel_voc)

def decoder_train_data(column, voc):
    train = []
    for word in column:
        indices = [voc["<go>"]]
        indices.extend(voc.get(char, voc["<unk>"]) for char in word.lower())
        indices.append(voc['<eos>'])
        train.append(indices)
    return train
y_traindecoder = decoder_train_data(dft['tel'],tel_voc)
y_valdecoder = decoder_train_data(dfdev['tel'],tel_voc)

# Pad the tokenized sequences to ensure consistent input lengths.
pad_length = 30
def padding(lists):
    for i in range(len((lists))):
        if len(lists[i])>pad_length:
            print('error')
        if len(lists[i]) < pad_length:
            pad_array = [eng_voc["<pad>"]]*(pad_length - len(lists[i]))
            lists[i].extend(pad_array)
    return lists
xtrainp = padding(xtrain)
ytrainp = padding(ytrain)
xvalp = padding(xval)
yvalp = padding(yval)
y_traindecoderp = padding(y_traindecoder)
y_valdecoderp = padding(y_valdecoder)

def batchgent(sourceseq, tarseq, batchsize):
    start = 0
    while start < len(sourceseq):
        end = min(start + batchsize, len(sourceseq))
        yield (torch.tensor(sourceseq[start:end]), torch.tensor(tarseq[start:end]))
        start = end