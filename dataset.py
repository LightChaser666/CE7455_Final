# -*- coding: utf-8 -*-
# @Author  : LI YI
# @Time    : 2022/4/16 20:42

from torch.utils.data import Dataset
import os


class DomainData(Dataset):
    def __init__(self, domain, split):
        super(Dataset, self).__init__()
        assert domain in ['laptop', 'movie', 'restaurant']
        assert split in ['train', 'test', 'val']
        path = './data/'
        if split == 'train':
            path += 'TrainingSens'
        elif split == 'test':
            path += 'TestSens'
        else:
            path += 'ValSens'
        data_x_file = os.path.join(path, domain + '_x.txt')
        data_y_file = os.path.join(path, domain + '_y.txt')
        self.x = []
        self.y = []
        max_sen_len = 40
        with open(data_x_file, mode='r') as fin:
            for line in fin:
                index = [int(i) for i in line.strip().split(' ')]
                if len(index) == max_sen_len:
                    self.x.append(index)
                else:
                    print('data Error!!!')
                    raise RuntimeError('dataError')
        with open(data_y_file, mode='r') as fin:
            for line in fin:
                index = int(line.strip().split(' ')[0])
                self.y.append(index)
        print('Dataset ' + domain + '-' + split + ' loaded.')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return item, self.x[item], self.y[item]
