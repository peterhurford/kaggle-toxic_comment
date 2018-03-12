import gc

import numpy as np
import pandas as pd

import pathos.multiprocessing as mp

from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from cv import run_cv_model
from utils import print_step
from preprocess import run_tfidf, clean_text
from cache import get_data, is_in_cache, load_cache, save_in_cache
from feature_engineering import add_features


# Combine both word-level and character-level
TFIDF_UNION1 = {'ngram_min': 1,
                'ngram_max': 1,
                'min_df': 1,
                'max_features': 10000,
                'rm_stopwords': True,
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
def runSparseLGB(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    print_step('Get K Best')
    model = LogisticRegression(solver='sag', max_iter=500)
    sfm = SelectFromModel(model, threshold=0.2)
    print(train_X.shape)
    train_sparse_matrix = sfm.fit_transform(train_X, train_y)
    print(train_sparse_matrix.shape)
    test_sparse_matrix = sfm.transform(test_X)
    test_sparse_matrix2 = sfm.transform(test_X2)

    print_step('Merging')
    train_fe, test_fe = load_cache('fe_lgb_data')
    train_X = train_fe.values[dev_index]
    test_X = train_fe.values[val_index]
    test_X2 = test_fe.values
    train_X = csr_matrix(hstack([csr_matrix(train_X), train_sparse_matrix]))
    print(train_X.shape)
    test_X = csr_matrix(hstack([csr_matrix(test_X), test_sparse_matrix]))
    test_X2 = csr_matrix(hstack([csr_matrix(test_X2), test_sparse_matrix2]))

    print_step('Garbage collection')
    del train_fe
    del test_fe
    del train_sparse_matrix
    del test_sparse_matrix
    gc.collect()

    print_step('Modeling')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
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
              'nthread': min(mp.cpu_count() - 1, 6),
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
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = get_data()


if not is_in_cache('cleaned'):
    print('~~~~~~~~~~~~~')
    print_step('Cleaning')
    train_cleaned, test_cleaned = clean_text(train, test)
    save_in_cache('cleaned', train_cleaned, test_cleaned)
else:
    train_cleaned, test_cleaned = load_cache('cleaned')
    print_step('Filling missing')
    train_cleaned['comment_text'].fillna('missing', inplace=True)
    test_cleaned['comment_text'].fillna('missing', inplace=True)
    print('Train shape: {}'.format(train_cleaned.shape))
    print('Test shape: {}'.format(test_cleaned.shape))



print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run TFIDF WORD-CHAR UNION')
if not is_in_cache('tfidf_char_union'):
    TFIDF_UNION1.update({'train': train, 'test': test})
    post_trainw, post_testw = run_tfidf(**TFIDF_UNION1)
    TFIDF_UNION2.update({'train': train, 'test': test})
    post_trainc, post_testc = run_tfidf(**TFIDF_UNION2)
    post_train = csr_matrix(hstack([post_trainw, post_trainc]))
    del post_trainw; del post_trainc; gc.collect()
    post_test = csr_matrix(hstack([post_testw, post_testc]))
    del post_testw; del post_testc; gc.collect()
    save_in_cache('tfidf_char_union', post_train, post_test)
    del post_train
    del post_test


print('~~~~~~~~~~')
print('Loading FE')
if not is_in_cache('fe_lgb_data'):
    print_step('Adding Features')
    train_fe, test_fe = add_features(train, test)
    print_step('Dropping')
    train_fe.drop(['id', 'comment_text'], axis=1, inplace=True)
    test_fe.drop(['id', 'comment_text'], axis=1, inplace=True)
    train_fe.drop(['toxic', 'severe_toxic', 'obscene', 'threat',
                   'insult', 'identity_hate'], axis=1, inplace=True)
    print_step('Saving')
    save_in_cache('fe_lgb_data', train_fe, test_fe)
    del train_fe
    del test_fe
    gc.collect()


print('~~~~~~~~~~~~')
print_step('Run LGB')
train, test = run_cv_model(label='sparse_fe_lgb',
                           data_key='tfidf_char_union',
                           model_fn=runSparseLGB,
                           train=train,
                           test=test,
                           kf=kf)
# toxic CV scores : [0.9826496062603199, 0.9830212932736853, 0.9815062563553301, 0.98022789149499, 0.981731541721145]
# toxic mean CV : 0.9818273178210941
# severe_toxic CV scores : [0.9907376375169112, 0.9888719942493184, 0.9903119467039991, 0.9922809301301098, 0.9887765464258907]
# severe_toxic mean CV : 0.9901958110052458
# obscene CV scores : [0.9933673973796135, 0.993919978856799, 0.9926754787135739, 0.9927263904855579, 0.9933408309332551]
# obscene mean CV : 0.9932060152737598
# threat CV scores : [0.9893472977361535, 0.9912972922362948, 0.9904282818441812, 0.99134220616599, 0.9881882482937496]
# threat mean CV : 0.9901206652552738
# insult CV scores : [0.9832124677272037, 0.9835326755212629, 0.9839356436291075, 0.986883748038697, 0.9858095196238779]
# insult mean CV : 0.9846748109080299
# identity_hate CV scores : [0.9843095304682539, 0.9885634545571751, 0.981675404744786, 0.9885268357417188, 0.988652610966542]
# identity_hate mean CV : 0.9863455672956952
# ('sparse_fe_lgb overall : ', 0.987728364593183)


import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 1')
save_in_cache('lvl1_sparse_fe_lgb', train, test)
print_step('Done!')

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['toxic'] = test['sparse_fe_lgb_toxic']
submission['severe_toxic'] = test['sparse_fe_lgb_severe_toxic']
submission['obscene'] = test['sparse_fe_lgb_obscene']
submission['threat'] = test['sparse_fe_lgb_threat']
submission['insult'] = test['sparse_fe_lgb_insult']
submission['identity_hate'] = test['sparse_fe_lgb_identity_hate']
submission.to_csv('submit/submit_sparse_fe_lgb.csv', index=False)
print_step('Done!')
