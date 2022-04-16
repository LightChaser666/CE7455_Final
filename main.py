# -*- coding: utf-8 -*-
# @Author  : LI YI
# @Time    : 2022/4/15 21:22

from data_helper import Loader, batch_index, load_word2id, load_y2id_id2y, load_word2vector, recover_data_from_files
from model import NetAB

import torch
from torch.utils.data import DataLoader

import argparse
import os



parser = argparse.ArgumentParser(description='NetAB for noisy label semantic classification')

parser.add_argument('--embedding_dim', type=int, default=300, help='dimensions of word embeddings')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',help='number of example per batch')
parser.add_argument('--n_hidden', type=int, default=300, help='number of hidden unit')
parser.add_argument('--lr', type=float, default=0.001,help='learning rate')
parser.add_argument('--n_class', type=int, default=2,help='number of distinct class')
parser.add_argument('--max_sentence_len', type=int, default=40,help='max number of words per sentence')
parser.add_argument('--max_doc_len', type=int, default=1,help='max number of sentences per doc')
parser.add_argument('--max_grad_norm', type=float, default=5.0,help='maximal gradient norm')
parser.add_argument('--l1_reg', type=float, default=0.001,help='l1 regularization')
parser.add_argument('--l2_reg', type=float, default=0.001,help='l2 regularization')
parser.add_argument('--random_base', type=float, default=0.01,help='initial random base')
parser.add_argument('--display_step', type=int, default=50,help='number of test display step')
parser.add_argument('--n_epoch', type=int, default=100,help='number of epoch')
parser.add_argument('--keep_prob1', type=float, default=0.5,help='dropout keep prob in data_layer')
parser.add_argument('--keep_prob2', type=float, default=1.0,help='dropout keep prob in softmax_layer')
parser.add_argument('--t1', type=str, default='last',help='type of hidden output')
parser.add_argument('--t2', type=str, default='last',help='type of hidden output')
parser.add_argument('--embedding_type', type=str, default='dynamic',help='embedding type: static or dynamic')
parser.add_argument('--model', type=str, default='NetAb',help='models: NetAb')
parser.add_argument('--decay_rate', type=float, default=0.96,help='decay rate of learning rate')

parser.add_argument('--early_stopping', type=int, default=5,help='the number of early stopping epoch')
parser.add_argument('--decay_steps', type=int, default=2000,help='decay rate of learning rate')

parser.add_argument('--is_train', type=bool, default=True,help='training or test')
parser.add_argument('--pre_trained', type=bool, default=True,help='whether has pre-trained embedding')

parser.add_argument('--ckpt_path', type=str, default='./ckpts_noisy/',help='the path of saving checkpoints')
parser.add_argument('--result_path', type=str, default='./results_noisy/',help='the path of saving results')
parser.add_argument('--word2id_path', type=str, default='./data/word2id/',help='the path of word2id')
parser.add_argument('--data_path', type=str, default='./data/',help='the path of dataset')
parser.add_argument('--gpu', type=str, default='0',help='choose to use which gpu')
parser.add_argument('--dataset', type=str, default='movie',help='movie, laptop, restaurant')

args = parser.parse_args([])


device = torch.device("cuda" if args.gpu else "cpu")

class Train(object):
    def __init__(self, domain, args, filter_list=(3,4,5), filter_num=100):
        
        # placeholder
        self.sen_x_batch = None
        self.sent_len_batch = None
        self.sen_y_batch = None
        self.keep_prob1 = None
        self.keep_prob2 = None



    def create_feed_dict(self, sen_x_batch, sent_len_batch, sen_y_batch, kp1=1.0, kp2=1.0):
        holder_list = [self.sen_x_batch, self.sent_len_batch, self.sen_y_batch,
                        self.keep_prob1, self.keep_prob2]
        feed_list = [sen_x_batch, sent_len_batch, sen_y_batch, kp1, kp2]
        return dict(zip(holder_list, feed_list))

def train():
    domain = args.dataset

    best_val_acc = 0
    best_val_epoch = 0
    # best_test_acc = 0
    training_path = os.path.join(args.data_path, 'TrainingSens/')
    train_sen_x, train_sen_len, train_sen_y = recover_data_from_files(
        training_path, 'training', domain, args.max_sentence_len)
    train_loader = DataLoader(
        Loader(train_sen_x, train_sen_y, device),batch_size=args.batch_size,shuffle=True)

    val_path = os.path.join(args.data_path, 'ValSens/')
    val_sen_x, val_sen_len, val_sen_y = recover_data_from_files(
        val_path, 'validation', domain, args.max_sentence_len)
    val_loader = DataLoader(
        Loader(val_sen_x, val_sen_y, device),batch_size=args.batch_size,shuffle=True)

    test_path = os.path.join(args.data_path, 'TestSens/')
    test_sen_x, test_sen_len, test_sen_y = recover_data_from_files(
        test_path, 'test', domain, args.max_sentence_len)
    test_loader = DataLoader(
        Loader(test_sen_x, test_sen_y, device),batch_size=args.batch_size,shuffle=True)
        
    trainer = Train(args.dataset, args)

    model = NetAB()
    model.to(device)

    for epoch_i in range(args.n_epoch):
        #print('=' * 20 + 'Epoch ', epoch_i, '=' * 20)
        total_loss = []
        total_acc_num = []
        total_num = []
        if epoch_i < 5:
            for batch, (data, targets) in enumerate(train_loader):

                logits = model(data)
        
                print(logits[0].shape, logits[1].shape)
                exit(0)

                feed_dict = trainer.create_feed_dict(train_sen_x[indices], train_sen_len[indices],
                                                        train_sen_y[indices],
                                                        args.keep_prob1, args.keep_prob2)
                
                logits = model(feed_dict)
'''
                classifier.pre_run(sess, feed_dict=feed_dict)
            continue
        for step, indices in enumerate(batch_index(len(train_sen_y), flags_.batch_size, n_iter=1), 1):
            indices = list(indices)
            # if epoch_i < 10:
            feed_dict = classifier.create_feed_dict(train_sen_x[indices], train_sen_len[indices],
                                                    train_sen_y[indices],
                                                    flags_.keep_prob1, flags_.keep_prob2)
            loss, acc_num, logits = classifier.run(sess, feed_dict=feed_dict)
            y_pred_set = np.argmax(logits, axis=1)
            y_true_set = np.argmax(train_sen_y[indices], axis=1)
            f_indices = np.arange(0, len(indices))
            valid_indices = f_indices[y_pred_set == y_true_set]
            indices_new = list(np.array(indices)[valid_indices])
            if indices_new is None:
                continue
            # else:
            #     indices_new = indices
            # indices_new = indices
            feed_dict = classifier.create_feed_dict(train_sen_x[indices_new], train_sen_len[indices_new],
                                                    train_sen_y[indices_new],
                                                    flags_.keep_prob1, flags_.keep_prob2)
            classifier.run_cleaner(sess, feed_dict=feed_dict)
            total_loss.append(loss)
            total_acc_num.append(acc_num)
            total_num.append(len(indices))
            verbose = flags_.display_step
            if step % verbose == 0:
                print('[INFO] Len {}, Epoch {} - Batch {} : loss = {}, acc = {}'.format(
                    len(indices_new), epoch_i, step, np.mean(total_loss[-verbose:]),
                    sum(total_acc_num[-verbose:]) * 1.0 / sum(total_num[-verbose:])))
        loss = np.mean(total_loss)
        acc = sum(total_acc_num) * 1.0 / sum(total_num)
        print('\n[INFO] Epoch {} : mean loss = {}, mean acc = {}'.format(epoch_i, loss, acc))
        if np.isnan(loss):
            raise ValueError('[Error] loss is not a number!')
        # validation
        val_acc, val_loss, val_f1 = test_case(sess, classifier, val_sen_x, val_sen_len, val_sen_y)
        print('[INFO] val loss: {}, val acc: {}, val f1: {}'.format(val_loss, val_acc, val_f1))
        # test
        test_acc, test_loss, test_f1 = test_case(sess, classifier, test_sen_x, test_sen_len, test_sen_y)
        print('[INFO] test loss: {}, test acc: {}, test f1: {}'.format(test_loss, test_acc, test_f1))
        print('=' * 25 + ' end', '=' * 25 + '\n')
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch_i
            # best_test_acc = test_acc
            if not os.path.exists(classifier.config.ckpt_path + classifier.config.model + '/'):
                os.makedirs(classifier.config.ckpt_path + classifier.config.model + '/')
            saver.save(sess, save_path=save_path)
        if epoch_i - best_val_epoch > classifier.config.early_stopping:
            # here early_stopping is 5 :> 'the number of early stopping epoch'
            print('Normal early stop at {}!'.format(best_val_epoch))
            # break
    
    print('Best val acc = {}'.format(best_val_acc))
    # print('Test acc = {}'.format(best_test_acc))
    best_val_epoch_save_path = classifier.config.result_path + classifier.config.model + '/'
    if not os.path.exists(best_val_epoch_save_path):
        os.makedirs(best_val_epoch_save_path)
    with open(best_val_epoch_save_path + domain + '_bestEpoch.txt', 'w', encoding='utf-8') as fin:
        fin.write('Best epoch: ' + str(best_val_epoch) + '\n')

    saver.restore(sess, save_path)
    print('Model restored from %s' % save_path)
    # # test now
    run_test(sess, classifier, domain, test_sen_x, test_sen_len, test_sen_y)
'''


if __name__ == '__main__':


    if args.is_train == True:
        train()
    



