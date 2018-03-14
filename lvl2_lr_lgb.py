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


if not is_in_cache('lvl1_lr_with_fe'):
    print_step('Importing Stage 2 Data')
    train, test = load_cache('lvl1_lr')

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
    save_in_cache('lvl1_lr_with_fe', train_without_targets, test_)
else:
    train, test = load_cache('lvl1_lr')
    train_, test_ = load_cache('lvl1_lr_with_fe')


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Level 2 LGB')
print(train_.columns.values)
train_, test_ = run_cv_model(label='lvl2_lgb',
                             data_key='lvl1_lr_with_fe',
                             model_fn=runLGB,
                             train=train,
                             test=test,
                             kf=kf)

import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 2')
save_in_cache('lvl2_lr_lgb', train, test)
print_step('Done!')

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['toxic'] = test_['lvl2_lgb_toxic']
submission['severe_toxic'] = test_['lvl2_lgb_severe_toxic']
submission['obscene'] = test_['lvl2_lgb_obscene']
submission['threat'] = test_['lvl2_lgb_threat']
submission['insult'] = test_['lvl2_lgb_insult']
submission['identity_hate'] = test_['lvl2_lgb_identity_hate']
submission.to_csv('submit/submit_lvl2_lgb2.csv', index=False)
print_step('Done')
# toxic CV scores : [0.9857907121500065, 0.9866052651064504, 0.9848811869427058, 0.9843181820885379, 0.9858359364083221]
# toxic mean CV : 0.9854862565392045
# severe_toxic CV scores : [0.9916725898004627, 0.9903101607779891, 0.9910998377287784, 0.9930997772057303, 0.9902714657144375]
# severe_toxic mean CV : 0.9912907662454795
# obscene CV scores : [0.9947307618893799, 0.9945103048634256, 0.9942631115677498, 0.9942161838548307, 0.9938664798006344]
# obscene mean CV : 0.9943173683952041
# threat CV scores : [0.9930639581172676, 0.994607192243628, 0.9936915312653865, 0.994345802276765, 0.9841264957604269]
# threat mean CV : 0.9919669959326948
# insult CV scores : [0.9864497548263004, 0.9874732778486353, 0.9871709052782918, 0.9890149459258527, 0.9885061952792192]
# insult mean CV : 0.98772301583166
# identity_hate CV scores : [0.9849420428535718, 0.9892833433439763, 0.9850967608604599, 0.989889156926868, 0.9899049069550212]
# identity_hate mean CV : 0.9878232421879793
# ('lvl2_lgb overall : ', 0.9897679408553705)
