import gc
import numpy as np

from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from utils import run_cv_model, print_step
from preprocess import get_data, run_tfidf


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
    return pred_test_y, pred_test_y2, model


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
    return pred_test_y, pred_test_y2, model


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = get_data()

print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

print('~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run TFIDF WORD STOP')
TFIDF_PARAMS_WORD_STOP.update({'train': train, 'test': test})
post_train, post_test = run_tfidf(**TFIDF_PARAMS_WORD_STOP)

print('~~~~~~~~~~~')
print_step('Run LR')
train, test = run_cv_model(label='tfidf_word_stop_lr',
                           train=train,
                           test=test,
                           post_train=post_train,
                           post_test=post_test,
                           model_fn=runLR,
                           kf=kf)
# Toxic:   0.97071085751838204
# Severe:  0.9858469322003458
# Obscene: 0.9834975327437192
# Threat:  0.98732987354980428
# Insult:  0.97758018514263978
# Hate:    0.97315936110261991
# 0.97968745704291849

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
TFIDF_PARAMS_CHAR.update({'train': train, 'test': test})
post_train, post_test = run_tfidf(**TFIDF_PARAMS_CHAR)

print('~~~~~~~~~~~')
print_step('Run LR')
train, test = run_cv_model(label='tfidf_char_lr',
                           train=train,
                           test=test,
                           post_train=post_train,
                           post_test=post_test,
                           model_fn=runLR,
                           kf=kf)
# Toxic:   0.9757378905615528
# Severe:  0.98809320098589093
# Obscene: 0.9876460330203326
# Threat:  0.98599830847433922
# Insult:  0.98180741213603251
# Hate:    0.98216254095868538
# 0.98357423102280561

print('~~~~~~~~~~~~~')
print_step('Run NBLR')
train, test = run_cv_model(label='tfidf_char_nblr',
                           train=train,
                           test=test,
                           post_train=post_train,
                           post_test=post_test,
                           model_fn=runNBLR,
                           kf=kf)
# Toxic:   0.98094395807405343
# Severe:  0.98588747442559532
# Obscene: 0.99171786294918685
# Threat:  0.98891238560837191
# Insult:  0.98347127302252113
# Hate:    0.98221178632244333
# 0.98552412340036188

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

print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 2')
train.to_csv('cache/train_lvl1.csv', index=False)
test.to_csv('cache/test_lvl1.csv', index=False)
