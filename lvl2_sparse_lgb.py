import pandas as pd

import pathos.multiprocessing as mp

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step
from cache import is_in_cache, load_cache, save_in_cache
from feature_engineering import add_features


def runLGB(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    rounds_lookup = {'toxic': 4700,
                     'severe_toxic': 1600,
                     'obscene': 3500,
                     'threat': 3100,
                     'insult': 2600,
                     'identity_hate': 1900}
    params = {
        'boosting': 'dart',
        'learning_rate': 0.01,
        'application': 'binary',
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'auc',
        'data_random_seed': 2,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.4,
        'nthread': min(mp.cpu_count() - 1, 6),
        'lambda_l1': 1,
        'lambda_l2': 1
    }
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=rounds_lookup[label],
                      valid_sets=watchlist,
                      verbose_eval=100)
    print(model.feature_importance())
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


if not is_in_cache('lvl1_sparse_lgb_with_fe'):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Importing Stage 2 Data')
    train, test = load_cache('lvl1_sparse_lgb')

    print('~~~~~~~~~~~~~~~~~~~')
    print('Feature engineering')
    train_, test_ = add_features(train, test)

    print('~~~~~~~~~~~~~')
    print_step('Dropping')
    cols_to_drop = ['id', 'comment_text']
    train_ = train_.drop(cols_to_drop, axis=1)
    test_ = test_.drop(cols_to_drop, axis=1)
    train_without_targets = train_.drop(['toxic', 'severe_toxic', 'obscene', 'threat',
                                         'insult', 'identity_hate'], axis=1)
    print('Train shape: {}'.format(train_without_targets.shape))
    print('Test shape: {}'.format(test_.shape))
    save_in_cache('lvl1_sparse_lgb_with_fe', train_without_targets, test_)
else:
    train, test = load_cache('lvl1_sparse_lgb')
    train_, test_ = load_cache('lvl1_sparse_lgb_with_fe')


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Level 2 LGB')
print(train_.columns.values)
train_, test_ = run_cv_model(label='lvl2_sparse_lgb',
                             data_key='lvl1_sparse_lgb_with_fe',
                             model_fn=runLGB,
                             train=train,
                             test=test,
                             kf=kf)


import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 2')
save_in_cache('lvl2_sparse_lgb', train, test)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['toxic'] = test_['lvl2_sparse_lgb_toxic']
submission['severe_toxic'] = test_['lvl2_sparse_lgb_severe_toxic']
submission['obscene'] = test_['lvl2_sparse_lgb_obscene']
submission['threat'] = test_['lvl2_sparse_lgb_threat']
submission['insult'] = test_['lvl2_sparse_lgb_insult']
submission['identity_hate'] = test_['lvl2_sparse_lgb_identity_hate']
submission.to_csv('submit/submit_lvl2_sparse_lgb.csv', index=False)
print_step('Done')
# toxic CV scores : [0.9832521744749525, 0.9833780375832, 0.9811525075864607, 0.9813690880029439, 0.9832414366548694]
# toxic mean CV : 0.9824786488604854
# severe_toxic CV scores : [0.9914146308746673, 0.9893902104465757, 0.9889458125244016, 0.9926309716280849, 0.9895329853092704]
# severe_toxic mean CV : 0.9903829221566
# obscene CV scores : [0.9934624830536264, 0.9936662604431262, 0.9933029043888473, 0.9933743042090458, 0.9932912602675237]
# obscene mean CV : 0.9934194424724339
# threat CV scores : [0.9886683271001604, 0.9877798380422598, 0.9896645217218224, 0.9941390341779397, 0.9793931604421198]
# threat mean CV : 0.9879289762968604
# insult CV scores : [0.984745493520511, 0.9854121565062676, 0.9858561011888411, 0.9879923007580478, 0.9867466334326377]
# insult mean CV : 0.9861505370812612
# identity_hate CV scores : [0.9841209298432674, 0.9846396162933142, 0.9830150008893142, 0.9866740699298999, 0.9884491543528635]
# identity_hate mean CV : 0.9853797542617319
# ('lvl2_sparse_lgb overall : ', 0.9876233801882289)
