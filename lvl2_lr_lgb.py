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
        'num_leaves': 61,         # consider upping this?
        'verbosity': -1,
        'metric': 'auc',
        'data_random_seed': 1,
        'bagging_fraction': 0.8,  # consider upping this to 1.0
        'feature_fraction': 0.2,
        'nthread': 4,
        'lambda_l1': 1,
        'lambda_l2': 1
    }
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=rounds_lookup[label],
                      valid_sets=watchlist,
                      verbose_eval=100) # 10
    print(model.feature_importance())
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Stage 2 Data')
train, test = get_stage2_data()


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
train_, test_ = run_cv_model(label='lvl2_lgb',
                             train=train,
                             test=test,
                             post_train=train_without_targets.values,
                             post_test=test_.values,
                             model_fn=runLGB,
                             kf=kf)


import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 2')
train.to_csv('cache/train_lgb_lr_lvl2.csv', index=False)
test.to_csv('cache/test_lgb_lr_lvl2.csv', index=False)

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

# Toxic: 0.9852689559384487
# Severe toxic: 0.9911143771337798
# Obscene: 0.9940067706939768
# Threat: 0.9917843413165744
# Insult: 0.987364092758875
# Identity Hate: 0.9870498467414743
# Overall: 0.9894313974305216



# import string
# from textblob import TextBlob
# from nltk.corpus import stopwords
# non_alphanums = re.compile(u'[^A-Za-z0-9]+')
# def tokenize_fn(text):
#     return u" ".join(
#         [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] if len(x) > 1]) if len(x) > 1])
# train['pos_space'] = train.comment_text.apply(lambda t: ' '.join([x[1] for x in TextBlob(tokenize_fn(t)).pos_tags]))
# test['pos_space'] = test.comment_text.apply(lambda t: ' '.join([x[1] for x in TextBlob(tokenize_fn(t)).pos_tags]))


# GOLD:    0.9871
# TOP 25:  0.9870
# TOP 50:  0.9866
# TOP 100: 0.9862
# SILVER:  0.9859
# BRONZE:  0.9845
