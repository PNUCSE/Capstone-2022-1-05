import os
import torch
from torch import nn

from transformers import BertTokenizer, AutoTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertForTokenClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel,AutoConfig

import pandas as pd
import numpy as np
import random
from konlpy.tag import Mecab

import nltk
from nltk import FreqDist

import subprocess

# files = ['output.txt', 'output2.txt']
files = ['test2.txt']

document = []
sentences = []
# tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
def mecab_tagger():
    # make pos_ids 42개
    tagger = [
        "NNG", # 일반 명사
        "NNP", # 고유 명사
        "NNB", # 의존 명사
        "NNBC", # 단위를 나타내는 명사
        "NR", # 수사
        "NP", # 대명사
        "VV", # 동사
        "VA", # 형용사
        "VX", # 보조 용언
        "VCP", # 긍정 지정사
        "VCN",# 부정 지정사
        "MM", # 관형사
        "MAG", # 일반 부사
        "MAJ", # 접속 부사
        "IC", # 감탄사
        "JKS", # 주격 조사
        "JKG", # 관형격 조사
        "JKO", # 목적격 조사
        "JKB", # 부사격 조사
        "JKV", # 호격 조사
        "JKQ", # 인용격 조사
        "JX", # 보조사
        "JC", # 접속 조사
        "EP", # 선어말 어미
        "EF", # 종결 어미
        "EC", # 연결 어미
        "ETN", # 명사형 전성 어미
        "ETM", # 관형형 전성 어미
        "XPN", # 체언 접두사
        "XSN", # 명사 파생 접미사
        "XSV", # 동사 파생 접미사
        "XSA", # 형용사 파생 접미사
        "XR",  # 어근
        'SF', # 마침표, 물음표, 느낌표
        "SE", # 줄임표
        "SSO", # 여는 괄호
        "SSC", # 닫는 괄호
        "SC", # 구분자
        "SY",
        "SL", # 외국어
        "SH", # 한자
        "SN" # 숫자
        ]

    pos_ids = {}
    for i, tag in enumerate(tagger):
        pos_ids[tag] = i + 3    # index 3부터 시작

    return pos_ids

# read file
print("파일 읽는중....")
for file_name in files:
    file = open(file_name, 'r', encoding='utf-8')
    sentence = file.read()
    file.close()

    sentences.append(sentence)

for s in sentences:
    docs = s.split('\n')
    for doc in docs:
        document.append(doc)

print("document length:", len(document))

MAX_LEN = 512
total_len = len(document)
input_ids = np.zeros(shape=[total_len, MAX_LEN], dtype="uint8")
input_masks = np.zeros(shape=[total_len, MAX_LEN], dtype="uint8")
input_segments = np.zeros(shape=[total_len, MAX_LEN], dtype=np.int32)
input_pos = np.zeros(shape=[total_len, MAX_LEN], dtype="uint8")

pos_ids = mecab_tagger()
pos_len = len(pos_ids) + 3

output_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
depend = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'XSN', 'XSV', 'XSA', 'VCP', 'VCN', 'NNBC']

# vocab 저장
print("vocab 저장중...")
for sentence in document:
    for word, info in mecab.pos(sentence):
        if '+' in info: # +붙은 태그
            plus_tag = info.split('+')[0]
            info = plus_tag
        w = word
        if info in depend:
            w = '##' + w
        output_tokens.append(w)
    if w in output_tokens:  # 중복X
        continue
    output_tokens.append(w)

vocab = FreqDist(np.hstack(output_tokens))
print("단어 집합의 크기: ", len(vocab))

output = open('./vocab.txt', 'w', encoding='utf-8')
for token in output_tokens:
    output.write(token + '\n')
output.close()

# vocab to index
voca_file = open('./vocab.txt', 'r', encoding='utf-8')
vocab = voca_file.read().split('\n')
word_to_index = {word: index for index, word in enumerate(vocab)}




tokenized = []
sum_unk = 0
print("문장 처리중...")
# sentence preprocessing
for i in range(len(document)):
    tokens = []
    pos = []
    for word, info in mecab.pos(document[i]):
        if '+' in info: # +가 붙은 경우
            plus_tag = info.split('+')[0]
            info = plus_tag
        if info not in pos_ids: # 형태소가 pos_ids에 없을 경우
            print(f'{info} is not exist')
            pos_ids[info] = pos_len
            pos_len += 1
        pos.append(pos_ids[info]) # 형태소 index 저장
        w = word
        if info in depend:
            w = '##' + w
        tokens.append(w)    # 토큰 저장

    pos.insert(0, 1) # [CLS] index = 1
    pos.append(2) # [SEP] index = 2

    tokens.insert(0, '[CLS]')
    tokens.append('[SEP]')
    token_ids = []

    # token to ids
    for t in tokens:
        try:
            token_ids.append(word_to_index[t])
        except KeyError:
            token_ids.append(word_to_index['[UNK]'])
            sum_unk += 1

    length = len(token_ids)
    if length > MAX_LEN:
        length = MAX_LEN

    for j in range(length):
        input_ids[i, j] = token_ids[j]
        input_masks[i, j] = 1

    length_pos = len(pos)
    if length_pos > MAX_LEN:
        length_pos = MAX_LEN

    for j in range(length_pos):
        input_pos[i, j] = pos[j]

print("token [unk] 개수:", sum_unk)

# convert to pytorch tensor
# torch_input_ids = torch.from_numpy(input_ids).type(torch.long)
# torch_input_mask = torch.from_numpy(input_masks).type(torch.long)
# # torch_input_segments = torch.from_numpy(input_segments).type(torch.long)
# torch_input_pos = torch.from_numpy(input_pos).type(torch.long)

# save npy
np.save('./corpus/npy/input_ids', input_ids)
np.save('./corpus/npy/input_mask', input_masks)
# np.save('./corpus/npy/input_segments', input_segments)
np.save('./corpus/npy/input_pos', input_pos)


# # add embedding layer
# model = BertModel.from_pretrained('bert-base-uncased')
# with torch.no_grad():
#     token_type_embeddings = torch.clone(model.embeddings.token_type_embeddings.weight)
#     print(token_type_embeddings.shape)