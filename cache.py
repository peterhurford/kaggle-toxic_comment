import os

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from utils import print_step


def get_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    print('Train shape: {}'.format(train.shape))
    print('Test shape: {}'.format(test.shape))

    print_step('Filling missing')
    train['comment_text'].fillna('missing', inplace=True)
    test['comment_text'].fillna('missing', inplace=True)
    print('Train shape: {}'.format(train.shape))
    print('Test shape: {}'.format(test.shape))
    return train, test


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def is_in_cache(key):
    train_path = 'cache/train_' + key + '.csv'
    test_path = 'cache/test_' + key + '.csv'
    if os.path.exists(train_path) and os.path.exists(test_path):
        return 'csv'
    else:
        train_path = 'cache/train_' + key + '.npcsr.npz'
        test_path = 'cache/test_' + key + '.npcsr.npz'
        if os.path.exists(train_path) and os.path.exists(test_path):
            return 'csr'
        else:
            if os.path.exists('cache/model_' + key + '.npy'):
                return 'dict'
            else:
                return False


def is_csr_matrix(matrix):
    return isinstance(matrix, csr_matrix)


def load_cache(key):
    if is_in_cache(key):
        if is_in_cache(key) == 'dict':
            train_path = 'cache/model_' + key + '.npy'
            train = np.load(train_path).tolist()
            test = None
        elif is_in_cache(key) == 'csr':
            train_path = 'cache/train_' + key + '.npcsr.npz'
            test_path = 'cache/test_' + key + '.npcsr.npz'
            train = load_sparse_csr(train_path)
            test = load_sparse_csr(test_path)
            print('Train shape: {}'.format(train.shape))
            print('Test shape: {}'.format(test.shape))
        else:
            train_path = 'cache/train_' + key + '.csv'
            test_path = 'cache/test_' + key + '.csv'
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            print('Train shape: {}'.format(train.shape))
            print('Test shape: {}'.format(test.shape))

        if test is None:
            print_step('Skipped... Loaded ' + train_path + ' from cache!')
        else:
            print_step('Skipped... Loaded ' + train_path + ' and ' + test_path + ' from cache!')
        return train, test
    else:
        raise ValueError


def save_in_cache(key, train, test):
    if isinstance(train, dict):
        train = np.array(train)
        train_path = 'cache/model_' + key
        np.save(train_path, train)
    elif is_csr_matrix(train):
        train_path = 'cache/train_' + key + '.npcsr'
        test_path = 'cache/test_' + key + '.npcsr'
        save_sparse_csr(train_path, train)
        save_sparse_csr(test_path, test)
    else:
        train_path = 'cache/train_' + key + '.csv'
        test_path = 'cache/test_' + key + '.csv'
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
    if test is None:
        print_step('Saved ' + train_path + ' to cache!')
    else:
        print_step('Saved ' + train_path + ' and ' + test_path + ' to cache!')
