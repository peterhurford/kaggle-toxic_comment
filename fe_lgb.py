import pandas as pd

import pathos.multiprocessing as mp

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step
from cache import get_data, is_in_cache, load_cache, save_in_cache
from feature_engineering import add_features


# LGB Model Definition
def runLGB(train_X, train_y, test_X, test_y, test_X2, label):
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


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = get_data()


print('~~~~~~~~~~~~~~~')
if not is_in_cache('fe_lgb_data'):
    ## Base Features
    print_step('Adding Features')
    train, test = add_features(train, test)

    # train['pos_space'] = train.comment_text.apply(lambda t: ' '.join([x[1] for x in TextBlob(tokenize_fn(t)).pos_tags]))
    # print_step('POS 1/2')
    # test['pos_space'] = test.comment_text.apply(lambda t: ' '.join([x[1] for x in TextBlob(tokenize_fn(t)).pos_tags]))
    # print_step('POS 2/2')

    print_step('Dropping')
    train.drop(['id', 'comment_text', 'afinn'], axis=1, inplace=True)
    test.drop(['id', 'comment_text', 'afinn'], axis=1, inplace=True)
    train.drop(['toxic', 'severe_toxic', 'obscene', 'threat',
                 'insult', 'identity_hate'], axis=1, inplace=True)
    print_step('Saving')
    save_in_cache('fe_lgb_data', train, test)


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
# toxic CV scores : [0.9637513738451069, 0.9642285094463718, 0.9608774786672482, 0.9598456713004437, 0.9611342950969638]
# toxic mean CV : 0.9619674656712268
# severe_toxic CV scores : [0.9890864027469054, 0.9876797398104239, 0.9878439457852395, 0.9892578534856067, 0.9850030832028202]
# severe_toxic mean CV : 0.9877742050061992
# obscene CV scores : [0.9903894949613599, 0.9902422266923127, 0.9877341687001356, 0.9896926322903385, 0.9869627190198718]
# obscene mean CV : 0.9890042483328039
# threat CV scores : [0.9821248048859695, 0.975311036644772, 0.9656288302586507, 0.9817650386573638, 0.9597594542645507]
# threat mean CV : 0.9729178329422613
# insult CV scores : [0.9735521182549207, 0.9721271669056747, 0.9724818545148813, 0.9783919382930557, 0.9777996059214333]
# insult mean CV : 0.9748705367779931
# identity_hate CV scores : [0.9696822667263949, 0.9747637298901671, 0.9622949388522032, 0.9762463700403865, 0.978523036609928]
# identity_hate mean CV : 0.972302068423816
# ('fe_lgb overall : ', 0.9764727261923833)
