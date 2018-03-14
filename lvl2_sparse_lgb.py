import gc
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
    print_step('Importing Stage 2 Data')
    train, test = load_cache('lvl1_sparse_lgb')

    print_step('Importing FE')
    train_fe, test_fe = load_cache('fe_lgb_data')

    print_step('Merging')
    train_ = pd.concat([train, train_fe], axis=1)
    test_ = pd.concat([test, test_fe], axis=1)
    del train_fe
    del test_fe
    gc.collect()

    print_step('Dropping')
    cols_to_drop = ['id', 'comment_text']
    train_ = train_.drop(cols_to_drop, axis=1)
    test_ = test_.drop(cols_to_drop, axis=1)
    train_without_targets = train_.drop(['toxic', 'severe_toxic', 'obscene', 'threat',
                                         'insult', 'identity_hate'], axis=1)
    print('Train shape: {}'.format(train_without_targets.shape))
    print('Test shape: {}'.format(test_.shape))
    print_step('Saving')
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
# toxic CV scores : [0.9843639954746378, 0.9846312761508647, 0.9822539782362569, 0.9821160904793381, 0.9840378342400983]
# toxic mean CV : 0.9834806349162392
# severe_toxic CV scores : [0.9914466773104488, 0.9882297058034163, 0.9905397514883957, 0.9926498230693023, 0.9895126455963776]
# severe_toxic mean CV : 0.990475720653588
# obscene CV scores : [0.9941909642180685, 0.9943814487987039, 0.9939025297502513, 0.9937348860265444, 0.9938407590797866]
# obscene mean CV : 0.9940101175746708
# threat CV scores : [0.9916847376305562, 0.9925270671611301, 0.992636409587563, 0.9951788295933119, 0.9855397970695171]
# threat mean CV : 0.9915133682084157
# insult CV scores : [0.9850902441999316, 0.9859317922367267, 0.9860101697380448, 0.9882561551818944, 0.987187277310555]
# insult mean CV : 0.9864951277334304
# identity_hate CV scores : [0.985477414386116, 0.9873521648919947, 0.9847925603167016, 0.988930317712943, 0.9892027931999929]
# identity_hate mean CV : 0.9871510501015497
# ('lvl2_sparse_lgb overall : ', 0.9888543365313156)

