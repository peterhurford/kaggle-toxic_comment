import gc
import pandas as pd

import pathos.multiprocessing as mp

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step
from cache import is_in_cache, load_cache, save_in_cache, get_data
from feature_engineering import add_features


def runLGB(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
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
    rounds_lookup = {'toxic': 4700,
                     'severe_toxic': 1600,
                     'obscene': 3500,
                     'threat': 3100,
                     'insult': 2600,
                     'identity_hate': 1900}
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=rounds_lookup[label],
                      valid_sets=watchlist,
                      verbose_eval=50)
    print(model.feature_importance())
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


if not is_in_cache('lvl2_all'):
    print_step('Importing 1/9: LRs')
    lr_train, lr_test = load_cache('lvl1_lr')
    print_step('Importing 2/9: FE')
    train_fe, test_fe = load_cache('fe_lgb_data')
    # print_step('Importing 3/9: ConvAI')
    # convai_train, convai_test = load_cache('convai_data')
    # print_step('Merging')
    # train_fe['id'] = lr_train['id']
    # test_fe['id'] = lr_train['id']
    # train_fe = pd.merge(train_fe, convai_train, on='id')
    # test_fe = pd.merge(test_fe, convai_test, on='id')
    # train_fe.drop(['id', 'TOXICITY', 'SEVERE_TOXICITY', 'identity_hate', 'insult',
    #                'obscene', 'severe_toxic', 'threat', 'toxic'], axis=1, inplace=True)
    # test_fe.drop(['id', 'TOXICITY', 'SEVERE_TOXICITY', 'identity_hate', 'insult',
    #                'obscene', 'severe_toxic', 'threat', 'toxic'], axis=1, inplace=True)
    print_step('Importing 4/9: Sparse LGBs')
    lgb_train, lgb_test = load_cache('lvl1_sparse_lgb')
    print_step('Importing 5/9: FM')
    fm_train, fm_test = load_cache('lvl1_fm')
    print_step('Importing 6/9: GRU')
    gru_train, gru_test = load_cache('lvl1_gru')
    print_step('Importing 7/9: GRU80')
    gru80_train, gru80_test = load_cache('lvl1_gru80')
    print_step('Importing 8/9: GRU-Conv')
    gru_conv_train, gru_conv_test = load_cache('lvl1_gru-conv')
    print_step('Importing 9/9: GRU128')
    gru128_train, gru128_test = load_cache('lvl1_gru128')

    print_step('Merging 1/2')
    lr_keep = [c for c in lr_train.columns if 'lr' in c]
    lgb_keep = [c for c in lgb_train.columns if 'lvl2' in c]
    fm_keep = [c for c in fm_train.columns if 'fm' in c]
    gru_keep = [c for c in gru_train.columns if 'gru' in c]
    gru80_keep = [c for c in gru80_train.columns if 'gru' in c]
    gru_conv_keep = [c for c in gru_conv_train.columns if 'gru' in c]
    gru128_keep = [c for c in gru128_train.columns if 'gru' in c]
    train_ = pd.concat([lr_train[lr_keep],
                        train_fe,
                        lgb_train[lgb_keep],
                        fm_train[fm_keep],
                        gru_train[gru_keep],
                        gru80_train[gru80_keep],
                        gru_conv_train[gru_conv_keep],
                        gru128_train[gru128_keep]], axis=1)
    print_step('Merging 2/2')
    test_ = pd.concat([lr_test[lr_keep],
                       test_fe,
                       lgb_test[lgb_keep],
                       fm_test[fm_keep],
                       gru_test[gru_keep],
                       gru80_test[gru80_keep],
                       gru_conv_test[gru_conv_keep],
                       gru128_test[gru128_keep]], axis=1)
    del lr_train
    del lr_test
    del train_fe
    del test_fe
    del lgb_train
    del lgb_test
    del fm_train
    del fm_test
    del gru_train
    del gru_test
    del gru80_train
    del gru80_test
    del gru128_train
    del gru128_test
    del gru_conv_train
    del gru_conv_test
    gc.collect()
    print('Train shape: {}'.format(train_.shape))
    print('Test shape: {}'.format(test_.shape))
    print_step('Saving')
    save_in_cache('lvl2_all', train_, test_)
else:
    train_, test_ = load_cache('lvl2_all')


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Level 2 LGB')
print(train_.columns.values)
train, test = get_data()
train_, test_ = run_cv_model(label='lvl2_final_lgb',
                             data_key='lvl2_all',
                             model_fn=runLGB,
                             train=train,
                             test=test,
                             kf=kf)

import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 2')
save_in_cache('lvl2_final_lgb', train, test)
print_step('Done!')

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['toxic'] = test_['lvl2_final_lgb_toxic']
submission['severe_toxic'] = test_['lvl2_final_lgb_severe_toxic']
submission['obscene'] = test_['lvl2_final_lgb_obscene']
submission['threat'] = test_['lvl2_final_lgb_threat']
submission['insult'] = test_['lvl2_final_lgb_insult']
submission['identity_hate'] = test_['lvl2_final_lgb_identity_hate']
submission.to_csv('submit/submit_lvl2_final_lgb.csv', index=False)
print_step('Done')
# toxic CV scores : [0.9882901654215092, 0.9887648879856854, 0.9880030287497278, 0.9869931547242587, 0.9885244086515887]
# toxic mean CV : 0.9881151291065541
# severe_toxic CV scores : [0.9926533297933431, 0.9909722928462253, 0.991772139653461, 0.9931960187740512, 0.9909165818765221]
# severe_toxic mean CV : 0.9919020725887207
# obscene CV scores : [0.9957652713132766, 0.9955844773664968, 0.9951517231495953, 0.9950048709282329, 0.9951136507172339]
# obscene mean CV : 0.995323998694967
# threat CV scores : [0.9920680908681396, 0.9946150491844494, 0.9942883313952879, 0.9938217692071023, 0.990475765124673]
# threat mean CV : 0.9930538011559304
# insult CV scores : [0.9885599828536521, 0.9893776886328315, 0.9895085847384869, 0.9910050712661215, 0.990214880599217]
# insult mean CV : 0.9897332416180618
# identity_hate CV scores : [0.9896027225987984, 0.9921043983866121, 0.9890155928653722, 0.9923731613670259, 0.9931091376825836]
# identity_hate mean CV : 0.9912410025800785
# ('lvl2_final_lgb overall : ', 0.9915615409573855)
