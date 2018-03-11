import gc
import numpy as np

import pathos.multiprocessing as mp

from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from cv import run_cv_model
from utils import print_step
from preprocess import run_tfidf, clean_text
from cache import get_data, is_in_cache, load_cache, save_in_cache


# TFIDF Hyperparams
TFIDF_PARAMS_WORD = {'ngram_min': 1,
                     'ngram_max': 2,
                     'min_df': 1,
                     'max_features': 200000,
                     'rm_stopwords': False,
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


# Sparse LGB Model Definition
def runSparseLGB(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    print_step('Get K Best')
    model = LogisticRegression(solver='sag')
    sfm = SelectFromModel(model, threshold=0.2)
    print(train_X.shape)
    train_sparse_matrix = sfm.fit_transform(train_X, train_y)
    print(train_sparse_matrix.shape)
    test_sparse_matrix = sfm.transform(test_X)
    test_sparse_matrix2 = sfm.transform(test_X2)
    del train_X
    del test_X
    del test_X2
    gc.collect()
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
    pred_test_y = model.predict(test_sparse_matrix)
    pred_test_y2 = model.predict(test_sparse_matrix2)
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


if not is_in_cache('tfidf_word'):
    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Run TFIDF WORD')
    TFIDF_PARAMS_WORD.update({'train': train, 'test': test})
    post_train, post_test = run_tfidf(**TFIDF_PARAMS_WORD)
    save_in_cache('tfidf_word', post_train, post_test)
    del post_train
    del post_test
    gc.collect()

if not is_in_cache('tfidf_word_cleaned'):
    TFIDF_PARAMS_WORD.update({'train': train_cleaned, 'test': test_cleaned})
    post_train_cleaned, post_test_cleaned = run_tfidf(**TFIDF_PARAMS_WORD)
    save_in_cache('tfidf_word_cleaned', post_train_cleaned, post_test_cleaned)
    del post_train_cleaned
    del post_test_cleaned
    gc.collect()
del train_cleaned
del test_cleaned
gc.collect()


print('~~~~~~~~~~~~')
print_step('Run LGB')
train, test = run_cv_model(label='tfidf_word_sparse_lgb',
                           data_key='tfidf_word',
                           model_fn=runSparseLGB,
                           train=train,
                           test=test,
                           kf=kf)
# toxic CV scores : [0.9702901662371838, 0.9696223100754018, 0.9678153536674818, 0.9676149003746513, 0.9711870679257228]
# toxic mean CV : 0.9693059596560882
# severe_toxic CV scores : [0.9801895978261603, 0.9726377779905455, 0.982170654159893, 0.9874727212204224, 0.9781001815195353]
# severe_toxic mean CV : 0.9801141865433113
# obscene CV scores : [0.9830528922626651, 0.9837422487164804, 0.9814396979867874, 0.9815393581964723, 0.9841164068501664]
# obscene mean CV : 0.9827781208025144
# threat CV scores : [0.9818704364268729, 0.9649259614276585, 0.9764352339273181, 0.9867757740570546, 0.9802860678000866]
# threat mean CV : 0.9780586947277982
# insult CV scores : [0.9750866289607637, 0.9725628946207349, 0.9733409509578796, 0.9770282683977928, 0.9761217897403539]
# insult mean CV : 0.974828106535505
# identity_hate CV scores : [0.9657562463199535, 0.9649726686386453, 0.9607666236203398, 0.968004717808433, 0.9670525723564731]
# identity_hate mean CV : 0.965310565748769
# ('tfidf_word_sparse_lgb overall : ', 0.9750659390023312)


print('~~~~~~~~~~~~~~~~~~~')
print_step('Run TFIDF CHAR')
if not is_in_cache('tfidf_char_cleaned'):
    TFIDF_PARAMS_CHAR.update({'train': train_cleaned, 'test': test_cleaned})
    post_train_cleaned, post_test_cleaned = run_tfidf(**TFIDF_PARAMS_CHAR)
    save_in_cache('tfidf_char_cleaned', post_train_cleaned, post_test_cleaned)
    del post_train_cleaned
    del post_test_cleaned
    gc.collect()


print('~~~~~~~~~~~~')
print_step('Run LGB')
train, test = run_cv_model(label='tfidf_char_sparse_lgb',
                           data_key='tfidf_char_cleaned',
                           model_fn=runSparseLGB,
                           train=train,
                           test=test,
                           kf=kf)
# toxic CV scores : [0.9789188979820485, 0.9793596964168235, 0.9780895946404702, 0.9768671337433639, 0.9790701777986253]
# toxic mean CV : 0.9784611001162663
# severe_toxic CV scores : [0.9875699515156278, 0.9839561832975239, 0.986522658192117, 0.9902668520722447, 0.9839592094499297]
# severe_toxic mean CV : 0.9864549709054886
# obscene CV scores : [0.9922752825211311, 0.9924230402161305, 0.9914373662844057, 0.9914883563671334, 0.9919979566033259]
# obscene mean CV : 0.9919244003984253
# threat CV scores : [0.9843666520003771, 0.9751319638685481, 0.9783999601914999, 0.9923843173840693, 0.9833505364391557]
# threat mean CV : 0.9827266859767299
# insult CV scores : [0.9815551484803782, 0.9816516155095475, 0.983102298524033, 0.9859975818227574, 0.9838976630492222]
# insult mean CV : 0.9832408614771877
# identity_hate CV scores : [0.9809779423328701, 0.9799994892490871, 0.9814785293928713, 0.985622811800776, 0.9835634956197484]
# identity_hate mean CV : 0.9823284536790705
# ('tfidf_char_sparse_lgb overall : ', 0.9841894120921947)


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
    gc.collect()


print('~~~~~~~~~~~~')
print_step('Run LGB')
train, test = run_cv_model(label='tfidf_union_sparse_lgb',
                           data_key='tfidf_char_union',
                           model_fn=runSparseLGB,
                           train=train,
                           test=test,
                           kf=kf)
# toxic CV scores : [0.9781458821170884, 0.9796751811907636, 0.9774164359237995, 0.9763362358568326, 0.979125692058316]
# toxic mean CV : 0.97813988542936
# severe_toxic CV scores : [0.9883306823092958, 0.9832857169079073, 0.98733957051456, 0.9908564060917936, 0.9851723493013308]
# severe_toxic mean CV : 0.9869969450249775
# obscene CV scores : [0.9919901331728017, 0.9926189378477984, 0.9918566322151603, 0.9916241276183199, 0.9918131317737762]
# obscene mean CV : 0.9919805925255712
# threat CV scores : [0.9842173701247683, 0.9855741655928847, 0.9882614030401122, 0.98693854190445, 0.9821430107420163]
# threat mean CV : 0.9854268982808463
# insult CV scores : [0.9812241384704781, 0.9817550783590435, 0.9828823396152575, 0.9852420243837232, 0.9833610946301301]
# insult mean CV : 0.9828929350917264
# identity_hate CV scores : [0.9814055420797074, 0.9839313712773262, 0.9766561520228718, 0.9866267073452393, 0.9847527914956148]
# identity_hate mean CV : 0.982674512844152
# ('tfidf_union_sparse_lgb overall : ', 0.9846852948661056)


import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 2')
save_in_cache('lvl1_sparse_lgb', train, test)
print_step('Done!')
