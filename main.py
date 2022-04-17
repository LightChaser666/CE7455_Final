# -*- coding: utf-8 -*-
# @Author  : LI YI
# @Time    : 2022/4/15 21:22


import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DomainData
from model import NetAB

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='NetAB for noisy label semantic classification')
    parser.add_argument('--embedding_dim', type=int, default=300, help='dimensions of word embeddings')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N', help='number of example per batch')
    parser.add_argument('--n_hidden', type=int, default=300, help='number of hidden unit')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--n_class', type=int, default=2, help='number of distinct class')
    parser.add_argument('--max_sentence_len', type=int, default=40, help='max number of words per sentence')
    parser.add_argument('--max_doc_len', type=int, default=1, help='max number of sentences per doc')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='maximal gradient norm')
    parser.add_argument('--l1_reg', type=float, default=0.001, help='l1 regularization')
    parser.add_argument('--l2_reg', type=float, default=0.001, help='l2 regularization')
    parser.add_argument('--random_base', type=float, default=0.01, help='initial random base')
    parser.add_argument('--display_step', type=int, default=100, help='number of test display step')
    parser.add_argument('--n_epoch', type=int, default=10, help='number of epoch')
    parser.add_argument('--keep_prob1', type=float, default=0.5, help='dropout keep prob in data_layer')
    parser.add_argument('--keep_prob2', type=float, default=1.0, help='dropout keep prob in softmax_layer')
    parser.add_argument('--t1', type=str, default='last', help='type of hidden output')
    parser.add_argument('--t2', type=str, default='last', help='type of hidden output')
    parser.add_argument('--embedding_type', type=str, default='dynamic', help='embedding type: static or dynamic')
    parser.add_argument('--model', type=str, default='NetAb', help='models: NetAb')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='decay rate of learning rate')
    parser.add_argument('--early_stopping', type=int, default=5, help='the number of early stopping epoch')
    parser.add_argument('--decay_steps', type=int, default=2000, help='decay rate of learning rate')
    parser.add_argument('--is_train', type=bool, default=True, help='training or test')
    parser.add_argument('--pre_trained', type=bool, default=True, help='whether has pre-trained embedding')
    parser.add_argument('--ckpt_path', type=str, default='./ckpts_noisy/', help='the path of saving checkpoints')
    parser.add_argument('--result_path', type=str, default='./results_noisy/', help='the path of saving results')
    parser.add_argument('--word2id_path', type=str, default='./data/word2id/', help='the path of word2id')
    parser.add_argument('--data_path', type=str, default='./data/', help='the path of dataset')
    parser.add_argument('--dataset', type=str, default='restaurant', help='movie, laptop, restaurant')
    return parser.parse_args()



def eval(model, loader):
    model.eval()
    
    total_loss = 0
    total_acc = 0
    total_num = 0
    with torch.no_grad():
        for idx, x, y in loader:
            x = torch.stack(x,dim=1).to(device)
            y = y.to(device)

            pred = model.pre_run(x)
            loss = ce(pred, y.to(device))
            y_pred = torch.argmax(pred, dim=1)
            indices_true = torch.arange(0, len(y)).to(device)
            indices_true = indices_true[y_pred == y]
            acc_num = len(indices_true)
            total_acc = total_acc + acc_num
            total_loss += loss.item() * y.shape[0]
            total_num = total_num + y.shape[0]

    print('[INFO] loss: {},  acc: {}'.format(total_loss / total_num, total_acc / total_num))
    
    return total_acc / total_num

if __name__ == '__main__':
    args = parse_args()
    domain = args.dataset
    epochs = args.n_epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NetAB(domain).to(device)
    # Dataset Preparation
    train_set = DomainData(domain, 'train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_set = DomainData(domain, 'test')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    val_set = DomainData(domain, 'val')
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    clip = args.max_grad_norm
    ce = torch.nn.CrossEntropyLoss()
    max_val_acc = 0
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    step_lr = torch.optim.lr_scheduler.StepLR(adam, args.decay_steps, args.decay_rate)
    early_stopping = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = []
        total_acc_num = []
        total_num = []

        loop = tqdm(train_loader)
        loop.set_description('Epoch %d' % epoch)
        if epoch <= 5:
            for idx, x, y in loop:
                adam.zero_grad()
                x = torch.stack(x,dim=1).to(device)
                y = y.to(device)
                pred = model.pre_run(x)
                y_pred = torch.argmax(pred, dim=1)
                indices_true = torch.arange(0, len(y)).to(device)
                indices_true = indices_true[y_pred == y]
                acc_num = len(indices_true)
                loss = ce(pred,y) + args.l2_reg * model.get_pre_l2()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                adam.step()
                step_lr.step()
                loop.set_postfix(acc= acc_num/ y.shape[0], loss=loss.item(), lr=step_lr.get_last_lr()[0])
            continue
        for idx, x, y in loop:
            model.train()
            adam.zero_grad()
            x = torch.stack(x, dim=1).to(device)
            y = y.to(device)
            noise, clean = model(x)
            loss = ce(noise, y) + args.l2_reg * model.get_l2()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            adam.step()
            step_lr.step()
            y_pred = torch.argmax(clean, dim=1)
            indices_true = torch.arange(0, len(y)).to(device)
            indices_true = indices_true[y_pred == y]
            y_nois = torch.argmax(noise, dim=1)
            indices_noise = torch.arange(0, len(y)).to(device)
            indices_noise = indices_noise[y_nois == y]
            acc_num = len(indices_noise)
            if len(indices_true) == 0:
                continue
            x_new = x[indices_true]
            y_new = y[indices_true]
            
            #  !!!!!!!!!!!!
            adam.zero_grad()
            #  !!!!!!!!!!!!            
            pred_new = model.pre_run(x_new)
            loss_new = ce(pred_new, y_new) + args.l2_reg * model.get_pre_l2()
            loss_new.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            adam.step()
            step_lr.step()

            total_loss.append(loss.detach().cpu())
            total_acc_num.append(acc_num)
            total_num.append(y.shape[0])

            loop.set_postfix(loss=loss.item(), lr=step_lr.get_last_lr()[0], acc=acc_num/y.shape[0])

        loss = np.mean(total_loss)
        acc = sum(total_acc_num) * 1.0 / sum(total_num)
        print('\n[INFO] Epoch {} : mean loss = {}, mean acc = {}'.format(epoch, loss, acc))
        print("=" * 50 + "val" + "=" * 50)
        val_acc = eval(model, val_loader)
        if val_acc > max_val_acc:
            torch.save(model.state_dict(),'./ckpt/'+domain+'.params')
            max_val_acc = val_acc
            early_stopping = 0
        else:
            early_stopping += 1
            if early_stopping >= 3:
                print('Early Stopping! ')
                break
            else:
                print('Early Stopping Count', early_stopping)
        print("=" * 50 + "test" + "=" * 50)
        eval(model, test_loader)
        print("\n")
    model.load_state_dict(torch.load('./ckpt/'+domain+'.params'))
    print('Final Test at best val acc: ' + str(max_val_acc))
    eval(model, test_loader)