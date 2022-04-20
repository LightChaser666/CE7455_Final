# -*- coding: utf-8 -*-
# @Author  : LI YI
# @Time    : 2022/4/17 21:40

# -*- coding: utf-8 -*-
# @Author  : LI YI
# @Time    : 2022/4/15 21:07

# file loading
import numpy as np
# pytorch
import torch
import torch.nn as nn


class NetTrans(nn.Module):
    def __init__(self, domain='movie', max_seq_len=40, dropout1=0.5, dropout2=1.0):
        super(NetTrans, self).__init__()
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
        self.transformer = TransformerModule(self.word_dim, self.max_seq_len, self.filter_num)
        self.noise_transformer = TransformerModule(self.word_dim, self.max_seq_len, self.filter_num)
        self.dropout2 = nn.Dropout(dropout2)
        d_hid = self.filter_num * 3
        self.clean_linear = nn.Linear(d_hid, 2)
        # Initialize with magic number
        nn.init.uniform_(self.clean_linear.weight,-0.01,0.01)
        nn.init.uniform_(self.clean_linear.bias,-0.01,0.01)
        self.transition1 = TransitionLayer(d_hid)
        self.transition2 = TransitionLayer(d_hid)

    def forward(self, x):
        mask = x==0
        x = self.embedding(x)
        x = self.dropout1(x).transpose(1,0)
        output1 = self.transformer(x, mask)
        output2 = self.noise_transformer(x, mask)
        clean_logits = self.clean_linear(output1)
        p = (self.transition1(output2).unsqueeze(2),
            self.transition2(output2).unsqueeze(2))
        prob = torch.cat(p, 2)
        sen_logits = clean_logits.unsqueeze(1)
        noisy_logits = sen_logits.matmul(prob).squeeze(1)
        return noisy_logits, clean_logits

    def pre_run(self, x):
        mask = x==0
        x = self.embedding(x)
        x = self.dropout1(x).transpose(1,0)
        output1 = self.transformer(x, mask)
        return self.clean_linear(output1)

    def get_pre_l2(self):
        return torch.norm(self.clean_linear.weight) + torch.norm(self.clean_linear.bias)

    def get_l2(self):
        return self.get_pre_l2() + self.transition1.get_l2() + self.transition2.get_l2()


class TransformerModule(nn.Module):
    def __init__(self, word_dim, max_seq_len, filter_num=100):
        super(TransformerModule, self).__init__()
        layer = nn.TransformerEncoderLayer(word_dim,3,300)
        self.encoder = nn.TransformerEncoder(layer,3)

    def forward(self, x, mask):
        return torch.mean(self.encoder(x,src_key_padding_mask=mask),dim=0)


class TransitionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(TransitionLayer, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        nn.init.uniform_(self.attn.weight,-0.01,0.01)
        nn.init.uniform_(self.attn.bias,0,0)
        self.tanh = nn.Tanh()
        self.u = nn.Linear(hidden_dim, 2,bias=False)
        nn.init.uniform_(self.u.weight,-0.01,0.01)

    def forward(self, x):
        z = self.tanh(self.attn(x))
        logit = self.u(z)
        e_logit = torch.exp(logit)
        e_sum = torch.sum(e_logit, dim=1, keepdim=True) + 1e-9
        return e_logit / e_sum

    def get_l2(self):
        return torch.norm(self.attn.weight)+torch.norm(self.attn.bias) + torch.norm(self.u.weight)


# model computation test
if __name__ == '__main__':
    model = NetTrans()
    model.to('cuda')
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 4, 7, 4, 8, 9, 10, 11, 12, 9, 13, 14, 15, 16, 17, 4, 12, 18, 16, 19, 20,
                       21, 13, 22, 23, 24, 9, 25, 0, 26, 27, 28, 29, 9, 30],
                      [9, 25, 38, 39, 40, 41, 42, 22, 43, 44, 4, 12, 18, 45, 9, 46, 22, 47, 48, 49, 9, 50, 26, 51, 52,
                       22, 53, 54, 0, 4, 1, 9, 55, 56, 57, 22, 58, 59, 60, 61]
                      ], dtype=torch.long).to('cuda')
    noise, clean = model(x)
    print('Success!')
