import numpy as np
import pandas as pd

import pathos.multiprocessing as mp

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, confusion_matrix

from cache import is_in_cache, load_cache, save_in_cache, get_data

from utils import print_step


def run_with_target(label, target, data_key, model_fn, kf, train_key=None, eval_fn=None):
    if is_in_cache(label + '_' + target):
        return load_cache(label + '_' + target)[0]
    else:
        print('-')
        print_step('Training ' + target)
        if train_key is None:
            train, test = get_data()
        else:
            train, test = load_cache(train_key)
        post_train, post_test = load_cache(data_key)
        if isinstance(post_train, pd.DataFrame):
            post_train = post_train.values
            post_test = post_test.values

        train_y = train[target]
        cv_scores = []
        pred_full_test = 0
        pred_train = np.zeros(train.shape[0])
        i = 1

        if isinstance(kf, StratifiedKFold):
            fold_splits = kf.split(post_train, train_y)
        else:
            fold_splits = kf.split(post_train)

        for dev_index, val_index in fold_splits:
            print_step('Started ' + label + ' ' + target + ' fold ' + str(i))
            dev_X, val_X = post_train[dev_index], post_train[val_index]
            dev_y, val_y = train_y[dev_index], train_y[val_index]
            pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, post_test, target, dev_index, val_index)
            pred_full_test = pred_full_test + pred_test_y
            pred_train[val_index] = pred_val_y
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(eval_fn(val_y, pred_val_y))
            print_step(label + ' ' + target + ' cv score ' + str(i) + ' : ' + str(cv_score))
            i += 1
        print_step(label + ' ' + target + ' cv scores : ' + str(cv_scores))
        mean_cv_score = np.mean(cv_scores)
        print_step(label + ' ' + target + ' mean cv score : ' + str(mean_cv_score))
        pred_full_test = pred_full_test / 5.
        results = {'label': label, 'target': target,
                   'train': pred_train, 'test': pred_full_test,
                    'cv': cv_scores}
        save_in_cache(label + '_' + target, results, None)
        return results


def run_cv_model(label, train, test, data_key, model_fn, kf, train_key=None, targets=None, eval_fn=None):
    mean_cv_scores = []
    actual_cv_scores = []

    if targets is None:
        targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    if eval_fn is None:
        eval_fn = roc_auc_score

    n_cpu = mp.cpu_count()
    n_nodes = min(n_cpu - 1, 6)
    print('Starting a jobs server with %d nodes' % n_nodes)
    pool = mp.ProcessingPool(n_nodes, maxtasksperchild=500)
    results = pool.map(lambda targ: run_with_target(label, targ, data_key, model_fn, kf, train_key, eval_fn), targets)
    for rr in results:
        print(rr['target'] + ' CV scores : ' + str(rr['cv']))
        mean_cv_score = np.mean(rr['cv'])
        actual_cv_score = eval_fn(train[rr['target']], rr['train'])
        print(rr['target'] + ' mean CV : ' + str(mean_cv_score) + ' overall: ' + str(actual_cv_score))
        mean_cv_scores.append(mean_cv_score)
        actual_cv_scores.append(actual_cv_score)
        train[rr['label'] + '_' + rr['target']] = rr['train']
        if test is not None:
            test[rr['label'] + '_' + rr['target']] = rr['test']
    pool.close()
    pool.join()
    pool.terminate()
    pool.restart()

    print(label + ' CV mean : ' + str(np.mean(mean_cv_scores)) + ', overall: ' + str(np.mean(actual_cv_scores)))
    return train, test
