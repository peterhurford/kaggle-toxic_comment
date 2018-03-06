import gc
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from utils import run_cv_model, print_step
from preprocess import get_data, run_tfidf


# TFIDF Hyperparams
TFIDF_UNION1 = {'ngram_min': 1,
                'ngram_max': 1,
                'min_df': 1,
                'max_features': 50000,
                'rm_stopwords': False,
                'analyzer': 'word',
                'token_pattern': r'\w{1,}',
                'sublinear_tf': True,
                'tokenize': True,
                'binary': False}
TFIDF_UNION2 = {'ngram_min': 2,
                'ngram_max': 6,
                'min_df': 1,
                'max_features': 50000,
                'rm_stopwords': True,
                'analyzer': 'char',
                'sublinear_tf': True,
                'tokenize': True,
                'binary': False}


# Sparse LGB Model Definition
def runSparseLGB(train_X, train_y, test_X, test_y, test_X2, label):
    print_step('Get K Best')
    model = LogisticRegression(solver='sag')
    sfm = SelectFromModel(model, threshold=0.2)
    train_sparse_matrix = sfm.fit_transform(train_X, train_y)
    test_sparse_matrix = sfm.transform(test_X)
    test_sparse_matrix2 = sfm.transform(test_X2)
    d_train = lgb.Dataset(train_sparse_matrix, label=train_y)
    d_valid = lgb.Dataset(test_sparse_matrix, label=test_y)
    watchlist = [d_train, d_valid]
    params = {'boosting': 'dart',
              'learning_rate': 0.1,
              'application': 'binary',
              'num_leaves': 31,
              'verbosity': -1,
              'metric': 'auc',
              'data_random_seed': 2,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.1,
              'nthread': 16,
              'lambda_l1': 1,
              'lambda_l2': 1,
              'min_data_in_leaf': 40}
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
    pred_test_y = model.predict(test_sparse_matrix)
    pred_test_y2 = model.predict(test_sparse_matrix2)
    return pred_test_y, pred_test_y2, model


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = get_data()


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run TFIDF WORD-CHAR UNION')
TFIDF_UNION1.update({'train': train, 'test': test})
post_trainw, post_testw = run_tfidf(**TFIDF_UNION1)
TFIDF_UNION2.update({'train': train, 'test': test})
post_trainc, post_testc = run_tfidf(**TFIDF_UNION2)
post_train = csr_matrix(hstack([post_trainw, post_trainc]))
del post_trainw; del post_trainc; gc.collect()
post_test = csr_matrix(hstack([post_testw, post_testc]))
del post_testw; del post_testc; gc.collect()


print_step('Run Sparse LGB')
train, test = run_cv_model(label='sparse_lgb',
                           train=train,
                           test=test,
                           post_train=post_train,
                           post_test=post_test,
                           model_fn=runSparseLGB,
                           kf=kf)


import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 1')
train.to_csv('cache/train_sparse_lgb_lvl1.csv', index=False)
test.to_csv('cache/test_sparse_lgb_lvl1.csv', index=False)


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['toxic'] = test['sparse_lgb_toxic']
submission['severe_toxic'] = test['sparse_lgb_severe_toxic']
submission['obscene'] = test['sparse_lgb_obscene']
submission['threat'] = test['sparse_lgb_threat']
submission['insult'] = test['sparse_lgb_insult']
submission['identity_hate'] = test['sparse_lgb_identity_hate']
submission.to_csv('submit/submit_sparse_lgb.csv', index=False)
print_step('Done')
# Toxic:   0.9785333073156789
# Severe:  0.9869978349294168
# Obscene: 0.9919044610414174
# Threat:  0.9852591206952639
# Insult:  0.9831517697662735
# Hate:    0.9828409344115459
# 0.9847812380265993
