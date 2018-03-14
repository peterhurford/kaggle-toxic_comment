import gc
import pandas as pd

import pathos.multiprocessing as mp

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step
from cache import is_in_cache, load_cache, save_in_cache, get_data
from feature_engineering import add_features


def runDoubleLGB(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    rounds_lookup = {'toxic': 1700,
                     'severe_toxic': 1000,
                     'obscene': 1200,
                     'threat': 1200,
                     'insult': 1200,
                     'identity_hate': 1200}
    params = {
        'learning_rate': 0.01,
        'application': 'binary',
        'num_leaves': 4,
        'verbosity': -1,
        'metric': 'auc',
        'data_random_seed': 12,
        'bagging_fraction': 1.0,
        'feature_fraction': 0.8,
        'nthread': min(mp.cpu_count() - 1, 6),
    }
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=rounds_lookup[label],
                      valid_sets=watchlist,
                      verbose_eval=50)
    print(model.feature_importance())
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


if not is_in_cache('lvl3'):
    print_step('Importing Stage 2 Data 1/6: LR LGB')
    lr_train, lr_test = load_cache('lvl2_lr_lgb')
    print_step('Importing Stage 2 Data 2/6: Sparse LGB')
    sparse_train, sparse_test = load_cache('lvl2_sparse_lgb')
    print_step('Importing Stage 2 Data 3/6: FM')
    fm_train, fm_test = load_cache('lvl1_fm')
    print_step('Importing Stage 2 Data 4/6: GRU')
    gru_train, gru_test = load_cache('lvl1_gru')
    print_step('Importing Stage 2 Data 5/6: GRU80')
    gru80_train, gru80_test = load_cache('lvl1_gru80')
    print_step('Importing Stage 2 Data 6/6: GRU-Conv')
    gru_conv_train, gru_conv_test = load_cache('lvl1_gru-conv')

    print_step('Merging 1/2')
    lr_keep = [c for c in lr_train.columns if 'lvl2' in c]
    sparse_keep = [c for c in sparse_train.columns if 'lvl2' in c]
    fm_keep = [c for c in fm_train.columns if 'fm' in c]
    gru_keep = [c for c in gru_train.columns if 'gru' in c]
    gru80_keep = [c for c in gru80_train.columns if 'gru' in c]
    gru_conv_keep = [c for c in gru_conv_train.columns if 'gru' in c]
    train_ = pd.concat([lr_train[lr_keep],
                        sparse_train[sparse_keep],
                        fm_train[fm_keep],
                        gru_train[gru_keep],
                        gru80_train[gru80_keep],
                        gru_conv_train[gru_conv_keep]], axis=1)
    print_step('Merging 2/2')
    test_ = pd.concat([lr_test[lr_keep],
                       sparse_test[sparse_keep],
                       fm_test[fm_keep],
                       gru_test[gru_keep],
                       gru80_test[gru80_keep],
                       gru_conv_test[gru_conv_keep]], axis=1)
    del lr_train
    del lr_test
    del sparse_train
    del sparse_test
    del fm_train
    del fm_test
    del gru_train
    del gru_test
    del gru80_train
    del gru80_test
    del gru_conv_train
    del gru_conv_test
    gc.collect()
    print('Train shape: {}'.format(train_.shape))
    print('Test shape: {}'.format(test_.shape))
    print_step('Saving')
    save_in_cache('lvl3', train_, test_)
else:
    train_, test_ = load_cache('lvl3')


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Level 2 LGB')
print(train_.columns.values)
train, test = get_data()
train_, test_ = run_cv_model(label='lvl3_lgb',
                             data_key='lvl3',
                             model_fn=runDoubleLGB,
                             train=train,
                             test=test,
                             kf=kf)

import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 3')
save_in_cache('lvl3_lgb', train, test)
print_step('Done!')

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['toxic'] = test_['lvl3_lgb_toxic']
submission['severe_toxic'] = test_['lvl3_lgb_severe_toxic']
submission['obscene'] = test_['lvl3_lgb_obscene']
submission['threat'] = test_['lvl3_lgb_threat']
submission['insult'] = test_['lvl3_lgb_insult']
submission['identity_hate'] = test_['lvl3_lgb_identity_hate']
submission.to_csv('submit/submit_lvl3_lgb.csv', index=False)
print_step('Done')
# toxic CV scores : [0.9871552223152595, 0.9879036716500451, 0.986876259984641, 0.9858631231480643, 0.9876899664874519]
# toxic mean CV : 0.9870976487170925
# severe_toxic CV scores : [0.9924240439942997, 0.9909047253121773, 0.9914948250313406, 0.9932860592103924, 0.9907909717471464]
# severe_toxic mean CV : 0.9917801250590713
# obscene CV scores : [0.9954073149602831, 0.995316252367598, 0.9951202128642624, 0.9948935326289543, 0.994886042906472]
# obscene mean CV : 0.9951246711455142
# threat CV scores : [0.9935702397414541, 0.9937520951842191, 0.9951440831997653, 0.9953506952370553, 0.9899634764830236]
# threat mean CV : 0.9935561179691035
# insult CV scores : [0.9882040848730831, 0.9890724972782181, 0.9891542919506925, 0.9905924743519918, 0.989995709712896]
# insult mean CV : 0.9894038116333764
# identity_hate CV scores : [0.98790857937662, 0.9912657656375561, 0.9892089244609524, 0.991919504306114, 0.9919462793539744]
# identity_hate mean CV : 0.9904498106270434
# ('lvl3_lgb overall : ', 0.991235364191867)

