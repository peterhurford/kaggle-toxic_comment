import pandas as pd

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from utils import run_cv_model, print_step
from preprocess import get_stage2_data
from feature_engineering import add_features


def runLGB(train_X, train_y, test_X, test_y, test_X2, label):
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    rounds_lookup = {'toxic': 160,
                     'severe_toxic': 120,
                     'obscene': 190,
                     'threat': 170,
                     'insult': 160,
                     'identity_hate': 140}
    params = {
        'learning_rate': 0.05,
        'application': 'binary',
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'auc',
        'data_random_seed': 1,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'nthread': 4,
        'lambda_l1': 1,
        'lambda_l2': 1
    }
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=rounds_lookup[label],
                      valid_sets=watchlist,
                      verbose_eval=10)
    print(model.feature_importance())
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Stage 2 Data')
train = pd.read_csv('cache/train_sparse_lgb_lvl1.csv')
test = pd.read_csv('cache/test_sparse_lgb_lvl1.csv')


print('~~~~~~~~~~~~~~~~~~~')
print('Feature engineering')
train, test = add_features(train, test)


print('~~~~~~~~~~~~~')
print_step('Dropping')
cols_to_drop = ['id', 'comment_text']
train_ = train.drop(cols_to_drop, axis=1)
test_ = test.drop(cols_to_drop, axis=1)
train_without_targets = train_.drop(['toxic', 'severe_toxic', 'obscene', 'threat',
                                     'insult', 'identity_hate'], axis=1)
print('Train shape: {}'.format(train_without_targets.shape))
print('Test shape: {}'.format(test.shape))


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Level 2 XGB')
print(train_without_targets.columns.values)
train_, test_ = run_cv_model(label='lvl2_sparse_lgb',
                             train=train,
                             test=test,
                             post_train=train_without_targets.values,
                             post_test=test_.values,
                             model_fn=runLGB,
                             kf=kf)


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
import pdb
pdb.set_trace()

# Toxic: 0.98078749659660169
# Severe toxic: 0.99016875165781426
# Obscene: 0.99268771563459224
# Threat: 0.98625341433375779
# Insult: 0.98504133284278106
# Identity Hate: 0.98469838473259164
# Overall: 0.98660618263302302
