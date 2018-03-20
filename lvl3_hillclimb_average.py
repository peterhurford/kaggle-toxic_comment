# Adapted from https://www.kaggle.com/tilii7/cross-validation-weighted-linear-blending-errors

import numpy as np

from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from cache import get_data, load_cache


def auc_func(weights):
    final_prediction = 0
    for weight, prediction in zip(weights, blend_train):
        final_prediction += weight * prediction
    return 1 - roc_auc_score(y_train, final_prediction)


base_train, base_test = get_data()
train, test = load_cache('lvl3_all_mix')
labels = ['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate']

for label in labels:
    y_train = base_train[label]
    print('\n Finding Blending Weights for ' + label + '...')
    blend_train = np.array([train['lvl2_all_lgb_' + label].rank().values,
                            train['lvl2_all_xgb_' + label].rank().values,
                            train['final-cnn_' + label].rank().values,
                            train['lvl2_all_rf_' + label].rank().values])
    blend_test = np.array([test['lvl2_all_lgb_' + label].rank().values,
                           test['lvl2_all_xgb_' + label].rank().values,
                           test['final-cnn_' + label].rank().values,
                           test['lvl2_all_rf_' + label].rank().values])

    res_list = []
    weights_list = []
    for k in range(2000):
        starting_values = np.random.uniform(size=len(blend_train))

        #######
        # I used to think that weights should not be negative - many agree with that.
        # I've come around on that issues as negative weights sometimes do help.
        # If you don't think so, just swap the two lines below.
        #######

    #   bounds = [(0, 1)]*len(blend_train)
        bounds = [(-1, 1)] * len(blend_train)

        res = minimize(auc_func,
                       starting_values,
                       method='L-BFGS-B',
                       bounds=bounds,
                       options={'disp': False,
                                'maxiter': 100000})
        res_list.append(res['fun'])
        weights_list.append(res['x'])
        print('{iter}\tScore: {score}\tWeights: {weights}'.format(
            iter=(k + 1),
            score=1 - res['fun'],
            weights='\t'.join([str(round(item, 6)) for item in res['x']])))

    bestSC = 1 - np.min(res_list)
    bestWght = weights_list[np.argmin(res_list)]
    weights = bestWght
    blend_score = 1 - round(bestSC, 6)

    print('\n Ensemble Score: {best_score}'.format(best_score=bestSC))
    print('\n Best Weights: {weights}'.format(weights=bestWght))

    train_score = np.zeros(len(blend_train[0]))
    test_score = np.zeros(len(blend_test[0]))

    print('\n Your final model:')
    for k in range(len(blend_test)):
        print(' %.6f * model-%d' % (weights[k], (k + 1)))
        test_score += blend_test[k] * weights[k]

    for k in range(len(blend_train)):
        train_score += blend_train[k] * weights[k]
