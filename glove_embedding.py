# -*- coding: utf-8 -*-
# @Author  : LI YI
# @Time    : 2022/4/15 16:22

import os
import pickle
from tqdm import tqdm
import numpy as np


def iter_count(path):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(path, 'r', encoding='utf-8') as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


def extract_embedding(path, output):
    print("Counting...")
    lines = iter_count(path)
    print("Word lines:", lines)
    id = 0
    word_embeddings = np.empty([lines, 300], dtype=float)
    with open(path, 'r', encoding='utf-8') as fout:
        for line in tqdm(fout, total=lines):
            line_s = line.split(' ')
            if len(line_s) != 300 + 1:
                print(u'a bad word embedding: {}'.format(line_s[0]))
                continue
            for i in range(300):
                word_embeddings[id, i] = float(line_s[i+1])
            id += 1
    np.save(output,word_embeddings)
    return word_embeddings


def extract_idx(path, output):
    word_idx = {}
    id = 0
    with open(path, 'r', encoding='utf-8') as fout:
        for line in tqdm(fout):
            line_s = line.split(' ')
            word = line_s[0]
            if word in word_idx:
                print("Duplicate word: " + word)
                print("Current id: " + str(id))
                print("Existing id: " + str(word_idx[word]))
                continue
            word_idx[word] = id
            id += 1
    with open(output, 'wb') as f:
        pickle.dump(word_idx, f)
    return word_idx


if __name__ == '__main__':
    word2vec_file = os.path.join('./data/glove.840B.300d.txt')
    word_idx_path = './data/word_idx.dict'
    word_embedding_path = './data/word_embeddings.npy'
    if not os.path.exists(word_idx_path):
        word_idx = extract_idx(word2vec_file, word_idx_path)
    else:
        with open(word_idx_path,'rb') as f:
            word_idx = pickle.load(f)
    if not os.path.exists(word_embedding_path):
        word_embeddings = extract_embedding(word2vec_file, word_embedding_path)
    else:
        word_embeddings = np.load(word_embedding_path)

    # Extract GloVe embeddings for the three datasets
    domains = ['movie', 'laptop', 'restaurant']
    for domain in domains:
        save_path = './data/Embeddings/' + domain + '.npy'
        word2id_file = os.path.join('./data/word2id', domain + '_w2id.txt')
        word2id = dict()
        with open(word2id_file, mode='r',encoding='utf-8') as fin:
            for line in fin:
                w2id = line.strip('\n').split('\t')
                word2id[w2id[0]] = int(w2id[1])
        w2v = np.empty([len(word2id),300],dtype=float)
        for word, id in tqdm(word2id.items()):
            if word in word_idx:
                w2v[id,:] = word_embeddings[word_idx[word],:]
            else:
                w2v[id,:] = np.random.uniform(-1, 1, [300])
        np.save(save_path,w2v)


