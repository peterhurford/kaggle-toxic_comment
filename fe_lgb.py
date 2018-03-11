import pandas as pd

import pathos.multiprocessing as mp

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step
from cache import get_data, is_in_cache, load_cache, save_in_cache
from feature_engineering import add_features


# LGB Model Definition
def runLGB(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    params = {'learning_rate': 0.05,
              'application': 'binary',
              'num_leaves': 31,
              'verbosity': -1,
              'metric': 'auc',
              'data_random_seed': 3,
              'bagging_fraction': 1.0,
              'feature_fraction': 0.4,
              'nthread': min(mp.cpu_count() - 1, 6),
              'lambda_l1': 1,
              'lambda_l2': 1}
    rounds_lookup = {'toxic': 1400,
                     'severe_toxic': 500,
                     'obscene': 550,
                     'threat': 380,
                     'insult': 500,
                     'identity_hate': 480}
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=rounds_lookup[label],
                      valid_sets=watchlist,
                      verbose_eval=10)
    print(model.feature_importance())
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~')
if not is_in_cache('fe_lgb_data'):
    print_step('Importing Data')
    train, test = get_data()
    print_step('Adding Features')
    train, test = add_features(train, test)
    print_step('Dropping')
    train.drop(['id', 'comment_text'], axis=1, inplace=True)
    test.drop(['id', 'comment_text'], axis=1, inplace=True)
    train.drop(['toxic', 'severe_toxic', 'obscene', 'threat',
                 'insult', 'identity_hate'], axis=1, inplace=True)
    print_step('Saving')
    save_in_cache('fe_lgb_data', train, test)
else:
    train, test = load_cache('fe_lgb_data')


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~~')
print_step('Run LGB')
print(train.columns.values)

train, test = get_data()
train, test = run_cv_model(label='fe_lgb',
                           data_key='fe_lgb_data',
                           model_fn=runLGB,
                           train=train,
                           test=test,
                           kf=kf)

import pdb
pdb.set_trace()
print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('lvl1_fe_lgb', train, test)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['toxic'] = test['fe_lgb_toxic']
submission['severe_toxic'] = test['fe_lgb_severe_toxic']
submission['obscene'] = test['fe_lgb_obscene']
submission['threat'] = test['fe_lgb_threat']
submission['insult'] = test['fe_lgb_insult']
submission['identity_hate'] = test['fe_lgb_identity_hate']
submission.to_csv('submit/submit_fe_lgb.csv', index=False)
print_step('Done!')
# toxic CV scores : [0.9716332989330163, 0.971155846124998, 0.9692958372138222, 0.9671419230498856, 0.9690416857295657]
# toxic mean CV : 0.9696537182102576
# severe_toxic CV scores : [0.988795554058071, 0.9865260812169696, 0.9889780584106946, 0.9905073071658793, 0.9845606696428792]
# severe_toxic mean CV : 0.9878735340988987
# obscene CV scores : [0.9903225414910851, 0.9911829816807867, 0.9895193501931143, 0.9907392260079375, 0.9890463030959521]
# obscene mean CV : 0.9901620804937752
# threat CV scores : [0.9877909687084236, 0.9826111167855683, 0.977648967597976, 0.9865405546678312, 0.968987431807881]
# threat mean CV : 0.9807158079135361
# insult CV scores : [0.9786169248574023, 0.9765427216447377, 0.9770361476165886, 0.9810249158058909, 0.9817663216647988]
# insult mean CV : 0.9789974063178837
# identity_hate CV scores : [0.9721974667105553, 0.9811971663899348, 0.9697375584058858, 0.9800579331035554, 0.9832834826192252]
# identity_hate mean CV : 0.9772947214458313
# ('fe_lgb2 overall : ', 0.9807828780800305)
