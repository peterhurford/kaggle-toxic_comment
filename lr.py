import gc
import numpy as np

from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from utils import run_cv_model, print_step
from preprocess import get_data, run_tfidf, clean_text


# TFIDF Hyperparams
TFIDF_PARAMS_WORD_STOP = {'ngram_min': 1,
                          'ngram_max': 2,
                          'min_df': 1,
                          'max_features': 200000,
                          'rm_stopwords': False,
                          'analyzer': 'word',
                          'tokenize': False,
                          'binary': True}

TFIDF_PARAMS_WORD_NOSTOP = {'ngram_min': 1,
                            'ngram_max': 2,
                            'min_df': 1,
                            'max_features': 200000,
                            'rm_stopwords': True,
                            'analyzer': 'word',
                            'tokenize': False,
                            'binary': True}

TFIDF_PARAMS_CHAR = {'ngram_min': 2,
                     'ngram_max': 6,
                     'min_df': 1,
                     'max_features': 200000,
                     'rm_stopwords': False,
                     'analyzer': 'char',
                     'tokenize': False,
                     'binary': True}

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


# LR Model Definition
def runLR(train_X, train_y, test_X, test_y, test_X2, label):
    model = LogisticRegression(solver='sag')
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)[:, 1]
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2


# Average both L1 and L2 Penalty with increased regularization
def runDoubleLR(train_X, train_y, test_X, test_y, test_X2, label):
    model = LogisticRegression(C=5, penalty='l1')
    model2 = LogisticRegression(C=5, penalty='l2')
    model.fit(train_X, train_y)
    model2.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)[:, 1] * 0.5 + model2.predict_proba(test_X)[:, 1] * 0.5
    pred_test_y2 = model.predict_proba(test_X2)[:, 1] * 0.5 + model2.predict_proba(test_X2)[:, 1] * 0.5
    return pred_test_y, pred_test_y2


# NB-LR Model Definition
# See https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline
def pr(x, y_i, y):
    p = x[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)

def runNBLR(train_X, train_y, test_X, test_y, test_X2, label):
    train_y = train_y.values
    r = csr_matrix(np.log(pr(train_X, 1, train_y) / pr(train_X, 0, train_y)))
    model = LogisticRegression(C=4, dual=True)
    x_nb = train_X.multiply(r)
    model.fit(x_nb, train_y)
    pred_test_y = model.predict_proba(test_X.multiply(r))[:, 1]
    pred_test_y2 = model.predict_proba(test_X2.multiply(r))[:, 1]
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = get_data()

print('~~~~~~~~~~~~~~~~~~~')
print_step('Cleaning')
train_cleaned, test_cleaned = clean_text(train, test)

print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

print('~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run TFIDF WORD STOP')
TFIDF_PARAMS_WORD_STOP.update({'train': train, 'test': test})
post_train, post_test = run_tfidf(**TFIDF_PARAMS_WORD_STOP)
TFIDF_PARAMS_WORD_STOP.update({'train': train_cleaned, 'test': test_cleaned})
post_train_cleaned, post_test_cleaned = run_tfidf(**TFIDF_PARAMS_WORD_STOP)

print('~~~~~~~~~~~')
print_step('Run LR')
train, test = run_cv_model(label='tfidf_word_stop_lr',
                           train=train,
                           test=test,
                           post_train=post_train_cleaned,
                           post_test=post_test_cleaned,
                           model_fn=runDoubleLR,
                           kf=kf)
# Toxic:   0.9737196334021132
# Severe:  0.98396212420271711
# Obscene: 0.98499225593799911
# Threat:  0.98840806131681003
# Insult:  0.9784074399165934
# Hate:    0.97444998683069495
# 0.98065658360115471

print('~~~~~~~~~~~~~')
print_step('Run NBLR')
train, test = run_cv_model(label='tfidf_word_stop_nblr',
                           train=train,
                           test=test,
                           post_train=post_train,
                           post_test=post_test,
                           model_fn=runNBLR,
                           kf=kf)
# Toxic:   0.97680262910046678
# Severe:  0.97712267525298491
# Obscene: 0.98730151563412805
# Threat:  0.98258078327396992
# Insult:  0.97848751250398303
# Hate:    0.96916136434636613
# 0.97857596470465602

print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run TFIDF WORD NOSTOP')
TFIDF_PARAMS_WORD_NOSTOP.update({'train': train, 'test': test})
post_train, post_test = run_tfidf(**TFIDF_PARAMS_WORD_NOSTOP)

print('~~~~~~~~~~~')
print_step('Run LR')
train, test = run_cv_model(label='tfidf_word_nostop_lr',
                           train=train,
                           test=test,
                           post_train=post_train,
                           post_test=post_test,
                           model_fn=runLR,
                           kf=kf)
# Toxic:   0.97259720529068228
# Severe:  0.98614096657718764
# Obscene: 0.98620193785468169
# Threat:  0.98435005925339603
# Insult:  0.97760170310553129
# Hate:    0.97610405043015458
# 0.98022614103723349

print('~~~~~~~~~~~~~')
print_step('Run NBLR')
train, test = run_cv_model(label='tfidf_word_nostop_nblr',
                           train=train,
                           test=test,
                           post_train=post_train,
                           post_test=post_test,
                           model_fn=runNBLR,
                           kf=kf)
# Toxic:   0.97249364318511444
# Severe:  0.97634936101993297
# Obscene: 0.98655480156746589
# Threat:  0.97426802020942049
# Insult:  0.97500133013082768
# Hate:    0.97013772753531524
# 0.97580081394134599

print('~~~~~~~~~~~~~~~~~~~')
print_step('Run TFIDF CHAR')
TFIDF_PARAMS_CHAR.update({'train': train_cleaned, 'test': test_cleaned})
post_train_cleaned, post_test_cleaned = run_tfidf(**TFIDF_PARAMS_CHAR)


print('~~~~~~~~~~~')
print_step('Run LR')
train, test = run_cv_model(label='tfidf_char_lr',
                           train=train,
                           test=test,
                           post_train=post_train_cleaned,
                           post_test=post_test_cleaned,
                           model_fn=runDoubleLR,
                           kf=kf)
# Toxic:   0.97942405783843223
# Severe:  0.98777339082449733
# Obscene: 0.99159022202176195
# Threat:  0.98942855482670455
# Insult:  0.98251217712541661
# Hate:    0.98453472009582277
# 0.98587718712210581

print('~~~~~~~~~~~~~')
print_step('Run NBLR')
train, test = run_cv_model(label='tfidf_char_nblr',
                           train=train,
                           test=test,
                           post_train=post_train_cleaned,
                           post_test=post_test_cleaned,
                           model_fn=runNBLR,
                           kf=kf)
# Toxic:   0.98117198787363669
# Severe:  0.98635829017675358
# Obscene: 0.99225265361509507
# Threat:  0.98889169152013889
# Insult:  0.98378758887481443
# Hate:    0.98240999840321253
# 0.98581203507727511

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

print('~~~~~~~~~~~')
print_step('Run LR')
train, test = run_cv_model(label='tfidf_union_lr',
                           train=train,
                           test=test,
                           post_train=post_train,
                           post_test=post_test,
                           model_fn=runLR,
                           kf=kf)
# Toxic:   0.97911493916630654
# Severe:  0.98851529710582897
# Obscene: 0.99058979906581368
# Threat:  0.99013879038260588
# Insult:  0.98296615331721193
# Hate:    0.98362711103100475
# 0.9858253483447954

print('~~~~~~~~~~~~~')
print_step('Run NBLR')
train, test = run_cv_model(label='tfidf_union_nblr',
                           train=train,
                           test=test,
                           post_train=post_train,
                           post_test=post_test,
                           model_fn=runNBLR,
                           kf=kf)
# Toxic:   0.97942436589251736
# Severe:  0.98566749783527785
# Obscene: 0.98993654381304419
# Threat:  0.98357519224915091
# Insult:  0.9816426907964898
# Hate:    0.97718255253247288
# 0.9829048071864922

import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 2')
train.to_csv('cache/train_lvl1.csv', index=False)
test.to_csv('cache/test_lvl1.csv', index=False)
print_step('Done!')
