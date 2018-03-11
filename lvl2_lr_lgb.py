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
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Importing Stage 2 Data')
    train, test = load_cache('lvl1_lr')

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
# toxic CV scores : [0.985816666459727, 0.986631525294112, 0.984804069042669, 0.9843376003463112, 0.9858273233967133]
# toxic mean CV : 0.9854834369079064
# severe_toxic CV scores : [0.991524164203159, 0.9902529119275549, 0.991006473485696, 0.9930275464204338, 0.9903163123009128]
# severe_toxic mean CV : 0.9912254816675514
# obscene CV scores : [0.994772226056059, 0.9944294908501818, 0.9942605468909068, 0.994165438493176, 0.9937996020085749]
# obscene mean CV : 0.9942854608597796
# threat CV scores : [0.992849856479881, 0.9935180238222446, 0.9929683653372723, 0.9938101901935681, 0.9824247115998558]
# threat mean CV : 0.9911142294865642
# insult CV scores : [0.986412548355863, 0.9873686647147805, 0.9870245066724844, 0.9889739907301461, 0.9884310210778666]
# insult mean CV : 0.9876421463102281
# identity_hate CV scores : [0.9858132168708068, 0.9889228926996707, 0.9858809997622869, 0.9896755190449902, 0.9897994942665961]
# identity_hate mean CV : 0.9880184245288701
# ('lvl2_lgb2 overall : ', 0.9896281966268167)


# GOLD:    0.9876S -> 0.9928CV
# TOP 25:  0.9874S -> 0.9926CV
# TOP 50:  0.9871S -> 0.9923CV
# TOP 100: 0.9869S -> 0.9921CV
# SILVER:  0.9865S -> 0.9917CV
# BRONZE:  0.9861S -> 0.9913CV
