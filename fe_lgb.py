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


if not is_in_cache('fe_lgb_data'):
    print('~~~~~~~~~~~~~~~')
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
# toxic CV scores : [0.9740811041477684, 0.9722730709683045, 0.97106642205402, 0.9686107884962571, 0.9716937000013259]
# toxic mean CV : 0.9715450171335351
# severe_toxic CV scores : [0.9898677206471516, 0.9878081776559822, 0.9899904800221851, 0.9908047630646689, 0.985505821374657]
# severe_toxic mean CV : 0.988795392552929
# obscene CV scores : [0.9913400188918419, 0.991934691000925, 0.9901329344445106, 0.9912267886956875, 0.9899785959722567]
# obscene mean CV : 0.9909226058010443
# threat CV scores : [0.9863721361450706, 0.986734046481662, 0.9853101396440281, 0.9896424069791676, 0.9748039011350741]
# threat mean CV : 0.9845725260770004
# insult CV scores : [0.9796309841766047, 0.9778976102194105, 0.977802692014103, 0.9825222603626638, 0.9829795730013279]
# insult mean CV : 0.9801666239548219
# identity_hate CV scores : [0.9724649837318602, 0.9829483445201658, 0.9723943631549242, 0.9808915033435621, 0.984380697080496]
# identity_hate mean CV : 0.9786159783662016
# ('fe_lgb overall : ', 0.9824363573142554)
