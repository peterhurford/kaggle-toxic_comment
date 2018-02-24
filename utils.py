from datetime import datetime

import numpy as np

from sklearn.metrics import roc_auc_score, confusion_matrix


def print_step(step):
    print('[{}]'.format(datetime.now()) + ' ' + step)


def run_cv_model(label, train, test, post_train, post_test, model_fn, kf):
    mean_cv_scores = []
    for target in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        print('-')
        print_step('Training ' + target)
        train_y = train[target]
        cv_scores = []
        pred_full_test = 0
        pred_train = np.zeros(train.shape[0])
        for dev_index, val_index in kf.split(post_train, train_y):
            dev_X, val_X = post_train[dev_index], post_train[val_index]
            dev_y, val_y = train_y[dev_index], train_y[val_index]
            pred_val_y, pred_test_y, model = model_fn(dev_X, dev_y, val_X, val_y, post_test, target)
            pred_full_test = pred_full_test + pred_test_y
            pred_train[val_index] = pred_val_y
            cv_score = roc_auc_score(val_y, pred_val_y)
            cv_scores.append(roc_auc_score(val_y, pred_val_y))
            print('cv score : ', cv_score)
        print('cv scores : ', cv_scores)
        mean_cv_score = np.mean(cv_scores)
        mean_cv_scores.append(mean_cv_score)
        print('Mean cv score : ', mean_cv_score)
        pred_full_test = pred_full_test / 5.
        print('Confusion matrix : TN    FP    FN    TP')
        print('Confusion matrix : {}'.format(confusion_matrix(train_y, pred_train > 0.5).ravel()))
        train[label + '_' + target] = pred_train
        test[label + '_' + target] = pred_full_test
    print_step('Done')
    print('Overall mean cv score : ', np.mean(mean_cv_scores))
    return train, test
