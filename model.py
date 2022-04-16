# -*- coding: utf-8 -*-
# @Author  : LI YI
# @Time    : 2022/4/15 21:07

# file loading
import numpy as np
# pytorch
import torch
import torch.nn as nn


class NetAB(nn.Module):
    def __init__(self, domain='movie', max_seq_len=40, dropout1=0.5, dropout2=1.0):
        super(NetAB, self).__init__()
        # Use GloVe as default embeddings
        print('Load GloVe Embedding for ' + domain + ' data...')
        assert domain in ['laptop', 'movie', 'restaurant']
        GloVe = np.load('./data/Embeddings/' + domain + '.npy')
        self.embedding = nn.Embedding(GloVe.shape[0], GloVe.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(GloVe))
        # Build model
        self.filter_num = 100
        self.max_seq_len = max_seq_len
        self.word_dim = self.embedding.weight.shape[1]
        self.dropout1 = nn.Dropout(dropout1)
        self.cnn = ConvModule(self.word_dim, self.max_seq_len, self.filter_num)
        self.noise_cnn = ConvModule(self.word_dim, self.max_seq_len, self.filter_num)
        self.dropout2 = nn.Dropout(dropout2)
        d_hid = self.filter_num * 3
        self.clean_linear = nn.Linear(d_hid, 2)

        self.transition1 = TransitionLayer(d_hid)
        self.transition2 = TransitionLayer(d_hid)

    def forward(self, x):

        x = self.embedding(x)
        x = self.dropout1(x)
        x = x.unsqueeze(1)
        output1 = self.cnn(x)
        output2 = self.noise_cnn(x)
        clean_logits = self.clean_linear(output1)
        p = (self.transition1(output2).unsqueeze(2),
            self.transition2(output2).unsqueeze(2))
        prob = torch.cat(p, 2)
        sen_logits = clean_logits.unsqueeze(1)
        noisy_logits = sen_logits.matmul(prob).squeeze(1)
        return noisy_logits, clean_logits


class ConvModule(nn.Module):
    def __init__(self, word_dim, max_seq_len, filter_num=100):
        super(ConvModule, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, filter_num, [3, word_dim], 1),
            nn.ReLU(),
            nn.MaxPool2d([max_seq_len - 3 + 1, 1], 1)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, filter_num, [4, word_dim], 1),
            nn.ReLU(),
            nn.MaxPool2d([max_seq_len - 4 + 1, 1], 1)
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, filter_num, [5, word_dim], 1),
            nn.ReLU(),
            nn.MaxPool2d([max_seq_len - 5 + 1, 1], 1)
        )

    def forward(self, x):        
        return torch.cat( (self.cnn1(x), self.cnn2(x), self.cnn3(x)) , dim=1).squeeze()



class TransitionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(TransitionLayer, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.u = nn.Linear(hidden_dim, 2,bias=False)

    def forward(self, x):
        z = self.tanh(self.attn(x))
        logit = self.u(z)
        e_logit = torch.exp(logit)
        e_sum = torch.sum(e_logit, dim=1, keepdim=True) + 1e-9
        return logit / e_sum


# model computation test
if __name__ == '__main__':
    model = NetAB()
    model.to('cuda')
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 4, 7, 4, 8, 9, 10, 11, 12, 9, 13, 14, 15, 16, 17, 4, 12, 18, 16, 19, 20,
                       21, 13, 22, 23, 24, 9, 25, 0, 26, 27, 28, 29, 9, 30],
                      [9, 25, 38, 39, 40, 41, 42, 22, 43, 44, 4, 12, 18, 45, 9, 46, 22, 47, 48, 49, 9, 50, 26, 51, 52,
                       22, 53, 54, 0, 4, 1, 9, 55, 56, 57, 22, 58, 59, 60, 61]
                      ], dtype=torch.long).to('cuda')
    noise, clean = model(x)
    print('Success!')
