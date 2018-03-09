import numpy as np
import pandas as pd

import pathos.multiprocessing as mp

from sklearn.metrics import roc_auc_score, confusion_matrix

from cache import is_in_cache, load_cache, save_in_cache, get_data

from utils import print_step


def run_with_target(label, target, data_key, model_fn, kf):
    if is_in_cache(label + '_' + target):
        return load_cache(label + '_' + target)[0]
    else:
        print('-')
        print_step('Training ' + target)
        train, test = get_data()
        post_train, post_test = load_cache(data_key)
        if isinstance(post_train, pd.DataFrame):
            post_train = post_train.values
            post_test = post_test.values

        train_y = train[target]
        cv_scores = []
        pred_full_test = 0
        pred_train = np.zeros(train.shape[0])
        i = 1
        for dev_index, val_index in kf.split(post_train, train_y):
            print_step('Started ' + label + ' ' + target + ' fold ' + str(i))
            dev_X, val_X = post_train[dev_index], post_train[val_index]
            dev_y, val_y = train_y[dev_index], train_y[val_index]
            pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, post_test, target)
            pred_full_test = pred_full_test + pred_test_y
            pred_train[val_index] = pred_val_y
            cv_score = roc_auc_score(val_y, pred_val_y)
            cv_scores.append(roc_auc_score(val_y, pred_val_y))
            print_step(label + ' ' + target + ' cv score ' + str(i) + ' : ' + str(cv_score))
            i += 1
        print_step(label + ' ' + target + ' cv scores : ' + str(cv_scores))
        mean_cv_score = np.mean(cv_scores)
        print_step(label + ' ' + target + ' mean cv score : ' + str(mean_cv_score))
        pred_full_test = pred_full_test / 5.
        print(label + ' ' + target + ' confusion matrix : TN    FP    FN    TP')
        print(label + ' ' + target + ' confusion matrix : {}'.format(confusion_matrix(train_y, pred_train > 0.5).ravel()))
        results = {'label': label, 'target': target,
                   'train': pred_train, 'test': pred_full_test,
                    'cv': cv_scores}
        save_in_cache(label + '_' + target, results, None)
        return results


def run_cv_model(label, train, test, data_key, model_fn, kf):
    mean_cv_scores = []
    targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    n_cpu = mp.cpu_count()
    n_nodes = min(n_cpu - 1, 3)
    print('Starting a jobs server with %d nodes' % n_nodes)
    pool = mp.ProcessingPool(n_nodes, maxtasksperchild=500)
    results = pool.map(lambda targ: run_with_target(label, targ, data_key, model_fn, kf), targets)
    for r in results:
        print(r['target'] + ' CV scores : ' + str(r['cv']))
        mean_cv_score = np.mean(r['cv'])
        print(r['target'] + ' mean CV : ' + str(mean_cv_score))
        mean_cv_scores.append(mean_cv_score)
        train[r['label'] + '_' + r['target']] = r['train']
        test[r['label'] + '_' + r['target']] = r['test']
    pool.close()
    pool.join()
    pool.terminate()
    pool.restart()

    print(label + ' overall : ', np.mean(mean_cv_scores))
    return train, test
