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
        'data_random_seed': 1,
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


if not is_in_cache('convai_with_fe'):
    print_step('Importing base data')
    train_base, test_base = get_data()

    print_step('Importing ConvAI data')
    train, test = load_cache('convai_data')

    print_step('Importing FE')
    train_fe, test_fe = load_cache('fe_lgb_data')

    print_step('Merging')
    train_fe['id'] = train_base['id']
    test_fe['id'] = test_base['id']
    train_ = pd.merge(train_fe, train, on='id')
    test_ = pd.merge(test_fe, test, on='id')
    del train_base
    del test_base
    del train_fe
    del test_fe
    gc.collect()

    print_step('Dropping')
    train_ = train_.drop('id', axis=1)
    test_ = test_.drop('id', axis=1)
    train_ = train_.drop(['toxic', 'severe_toxic', 'obscene', 'threat',
                          'insult', 'identity_hate'], axis=1)
    test_ = test_.drop(['toxic', 'severe_toxic', 'obscene', 'threat',
                        'insult', 'identity_hate'], axis=1)
    print('Train shape: {}'.format(train_.shape))
    print('Test shape: {}'.format(test_.shape))
    print_step('Saving')
    save_in_cache('convai_with_fe', train_, test_)
else:
    train_, test_ = load_cache('convai_with_fe')


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~~~~~~~~~')
print_step('Run ConvAI LGB')
print(train_.columns.values)
train, test = get_data()
train_, test_ = run_cv_model(label='convai_lgb',
                             data_key='convai_with_fe',
                             model_fn=runLGB,
                             train=train,
                             test=test,
                             kf=kf)

import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 2')
save_in_cache('lvl2_convai_lgb', train, test)
print_step('Done!')

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['toxic'] = test_['convai_lgb_toxic']
submission['severe_toxic'] = test_['convai_lgb_severe_toxic']
submission['obscene'] = test_['convai_lgb_obscene']
submission['threat'] = test_['convai_lgb_threat']
submission['insult'] = test_['convai_lgb_insult']
submission['identity_hate'] = test_['convai_lgb_identity_hate']
submission.to_csv('submit/submit_convai_lgb.csv', index=False)
print_step('Done')
# toxic CV scores : [0.9896037185875817, 0.9900950038758134, 0.9896205446979914, 0.9888497112383846, 0.9895100205023675]
# toxic mean CV : 0.9895357997804277
# severe_toxic CV scores : [0.9927467902964582, 0.9913065090553891, 0.9917506093232282, 0.9932600144560788, 0.9909299763215977]
# severe_toxic mean CV : 0.9919987798905504
# obscene CV scores : [0.9955709887872515, 0.9951898434326379, 0.9946437017801599, 0.9944661321697401, 0.9945177449074932]
# obscene mean CV : 0.9948776822154566
# threat CV scores : [0.990708512733482, 0.9913986140356392, 0.9934865960589584, 0.9938204458912697, 0.9833641004264386]
# threat mean CV : 0.9905556538291576
# insult CV scores : [0.9884591552010504, 0.988732975164064, 0.9889282849828682, 0.9907487298291214, 0.9893474787238383]
# insult mean CV : 0.9892433247801886
# identity_hate CV scores : [0.9879304599740314, 0.9870737268942867, 0.9875444277356646, 0.9904356829037831, 0.9904092453565261]
# identity_hate mean CV : 0.9886787085728583
# ('convai_lgb overall : ', 0.9908149915114399)
