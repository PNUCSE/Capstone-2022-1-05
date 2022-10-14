import os
import torch
from torch import nn

from functools import reduce
import operator

import pandas as pd
import numpy as np
from konlpy.tag import Mecab
from transformers import RobertaTokenizer
import nltk
from nltk import FreqDist

# vocab : 85293
# total_len : 150081

# files = ['./corpus/every_corpus_preprocessing_dev.txt','./corpus/every_corpus_preprocessing_test.txt','./corpus/every_corpus_preprocessing_train.txt']
files = ['./corpus/every_corpus_preprocessing_dev.txt']
tagger = [
        "NNG",
        "NNP",
        "NNB",
        "NP",
        "NR",
        "VV",
        "VA",
        "VX",
        "VCP",
        "VCN",
        "MMA",
        "MMD",
        "MMN",
        "MAG",
        "MAJ",
        "JC",
        "IC",
        "JKS",
        "JKC",
        "JKG",
        "JKO",
        "JKB",
        "JKV",
        "JKQ",
        "JX",
        "EP",
        "EF",
        "EC",
        "ETN",
        "ETM",
        "XPN",
        "XSN",
        "XSV",
        "XSA",
        "XR",
        "SF",
        "SP",
        "SS",
        "SE",
        "SO",
        "SL",
        "SH",
        "SW",
        "SN",
        "NA",
        "NF",
        "NV",
    ]

dependent = ['EP', 'EF', 'EC', 'ETN', 'ETM', 'XSN', 'XSV', 'XSA', 'VCP', 'VCN', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'NNB']
tag = {}
for i, t in enumerate(tagger):
    tag[t] = i + 3      # CLS:1, SEP:2

sentences = []
tokens = []
print("sentence 합치는 중...")
for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "" or line == "\n" or line == "\t":
                continue
            if line.startswith("#"):
                pos_ids = []
                token = []
                parsed = line.strip().split("\t")
                if len(parsed) != 2:  # metadata line about dataset
                    continue
                else:
                    # sent_id += 1
                    text = parsed[1].strip()
                    guid = parsed[0].replace("##", "").strip()
                    sentences.append(tokens)
                    tokens = []

            else:
                token_list = [token.replace("\n", "") for token in line.split("\t")] # lemma : 2, pos : 3
                tokens.append((token_list[2], token_list[3]))

sentences.pop(0)

# vocab 저장
print("vocab 저장 중...")
output_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
for sentence in sentences:
    for word, info in sentence:
        word_list = word.split()
        if '+' in info: # +붙은 태그
            info_list = info.split('+')
            for index, i in enumerate(info_list):
                if i in dependent:
                    word_list[index] = "##" + word_list[index]
        else:
            if info in dependent:
                word_list[0] = "##" + word_list[0]

        for w in word_list:
            if w in output_tokens: # 중복X
                continue
            output_tokens.append(w)

print("단어 집합의 크기: ", len(output_tokens))

output = open('./corpus/vocab.txt', 'w', encoding='utf-8')
for token in output_tokens:
    output.write(token + '\n')
output.close()

# vocab to index
voca_file = open('./corpus/vocab.txt', 'r', encoding='utf-8')
vocab = voca_file.read().split('\n')
word_to_index = {word: index for index, word in enumerate(vocab)}


# sentence preprocessing
MAX_LEN = 512
total_len = len(sentences)
input_ids = np.zeros(shape=[total_len, MAX_LEN], dtype="int32")
input_masks = np.zeros(shape=[total_len, MAX_LEN], dtype="int32")
input_pos = np.zeros(shape=[total_len, MAX_LEN], dtype="int32")
print("total_len: ", total_len)

sum_unk = 0
for i, sentence in enumerate(sentences):
    pos_ids = []
    token = []
    # pos_ids, token 생성
    for lemma, pos in sentence:
        lemma_list = lemma.split() # lemma split
        if '+' in pos:              # pos에 +태그가 있을 경우 +기준으로 split
            pos_list = pos.split('+')
            for index, p in enumerate(pos_list):
                pos_ids.append(tag[p])
                if p in dependent:  # 의존되는 경우 ##
                    lemma_list[index] = "##" + lemma_list[index]
        else:
            pos_ids.append(tag[pos])
            if pos in dependent:
                lemma_list[0] = "##" + lemma_list[0]
        token.append(lemma_list)
    token_list = list(reduce(operator.add, token))
    token_list.insert(0, '[CLS]')
    token_list.append('[SEP]')
    pos_ids.insert(0, 1) # CLS
    pos_ids.append(2)   # SEP

    # token_ids 생성
    token_ids = []
    for t in token_list:
        try:
            token_ids.append(word_to_index[t])
        except KeyError:
            token_ids.append(word_to_index['[UNK]'])
            sum_unk += 1

    length = len(pos_ids)
    if length > MAX_LEN:
        length = MAX_LEN

    for j in range(length):
        input_ids[i, j] = token_ids[j]
        input_masks[i, j] = 1
        input_pos[i, j] = pos_ids[j]

print("token [unk] 개수:", sum_unk)

# npy 저장
if not os.path.isdir('./corpus/npy'):
    os.mkdir('./corpus/npy')
np.save('./corpus/npy/input_ids', input_ids)
np.save('./corpus/npy/input_mask', input_masks)
np.save('./corpus/npy/input_pos', input_pos)
