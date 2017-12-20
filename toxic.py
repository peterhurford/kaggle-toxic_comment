# Basic architecture is to use TFIDF on both the word level and character level to extract info
# from the text. However, this creates too many features for an XGB to do well. So we use
# level 1 models (currently Naive Bayes and LR) plus SVD to extract the info for XGB, along with
# custom feature engineering.

# Design adapted from https://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author


# Libraries
import string

from datetime import datetime

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, confusion_matrix

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# TFIDF Hyperparams
MIN_DF = 5
MAX_FEATURES = 40000


# Utilities
def print_step(step):
    print('[{}]'.format(datetime.now()) + ' ' + step)

eng_stopwords = set(stopwords.words('english'))

def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 3
    param['silent'] = 1
    param['eval_metric'] = 'mlogloss'
    param['min_child_weight'] = child
    param['subsample'] = 0.8
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 2000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=50)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model


def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model


def runLR(train_X, train_y, test_X, test_y, test_X2):
    model = LogisticRegression()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model


# Let the modeling begin! :D
print_step('Reading data')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print('Train shape: {}'.format(train.shape))
print('Test shape: {}'.format(test.shape))


print_step('Filling missing')
train['comment_text'].fillna('missing', inplace=True)
test['comment_text'].fillna('missing', inplace=True)


print_step('Feature engineering 1/8')
train['num_words'] = train['comment_text'].apply(lambda x: len(str(x).split()))
test['num_words'] = test['comment_text'].apply(lambda x: len(str(x).split()))
print_step('Feature engineering 2/8')
train['num_unique_words'] = train['comment_text'].apply(lambda x: len(set(str(x).split())))
test['num_unique_words'] = test['comment_text'].apply(lambda x: len(set(str(x).split())))
print_step('Feature engineering 3/8')
train['num_chars'] = train['comment_text'].apply(lambda x: len(str(x)))
test['num_chars'] = test['comment_text'].apply(lambda x: len(str(x)))
print_step('Feature engineering 4/8')
train['num_stopwords'] = train['comment_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test['num_stopwords'] = test['comment_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
print_step('Feature engineering 5/8')
train['num_punctuations'] = train['comment_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test['num_punctuations'] = test['comment_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
print_step('Feature engineering 6/8')
train['num_words_upper'] = train['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test['num_words_upper'] = test['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
print_step('Feature engineering 7/8')
train['num_words_title'] = train['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test['num_words_title'] = test['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
print_step('Feature engineering 8/8')
train['mean_word_len'] = train['comment_text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test['mean_word_len'] = test['comment_text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
print('Train shape: {}'.format(train.shape))
print('Test shape: {}'.format(test.shape))


print_step('Dropping')
targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
cols_to_drop = ['id', 'comment_text']
train_X = train.drop(cols_to_drop + targets, axis=1)
test_X = test.drop(cols_to_drop, axis=1)
print('Train shape: {}'.format(train.shape))
print('Test shape: {}'.format(test.shape))


print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print_step('TFIDF Word 1/3')
tfidf_vec = TfidfVectorizer(ngram_range=(1, 3),
                            stop_words='english',
                            min_df=MIN_DF,
                            max_features=MAX_FEATURES)
full_tfidf = tfidf_vec.fit_transform(train['comment_text'].values.tolist() + test['comment_text'].values.tolist())
print_step('TFIDF Word 2/3')
train_tfidf = tfidf_vec.transform(train['comment_text'].values.tolist())
print_step('TFIDF Word 3/3')
test_tfidf = tfidf_vec.transform(test['comment_text'].values.tolist())
print('TFIDF Word train shape: {}'.format(train_tfidf.shape))
print('TFIDF Word test shape: {}'.format(test_tfidf.shape))


print_step('MNB TFIDF Word')
for target in targets:
    print(target)
    train_y = train[target]
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros(train.shape[0])
    for dev_index, val_index in kf.split(train_X, train_y):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_val_y = pred_val_y[:, 1]
        pred_test_y = pred_test_y[:, 1]
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        cv_scores.append(log_loss(val_y, pred_val_y))
    print('cv scores : ', cv_scores)
    print('Mean cv score : ', np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    print('Confusion matrix : {}'.format(confusion_matrix(train_y, pred_train > 0.5).ravel()))
    train['word_tfidf_mnb_' + target] = pred_train
    test['word_tfidf_mnb_' + target] = pred_full_test


print_step('LR TFIDF Word')
for target in targets:
    print(target)
    train_y = train[target]
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros(train.shape[0])
    for dev_index, val_index in kf.split(train_X, train_y):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runLR(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_val_y = pred_val_y[:, 1]
        pred_test_y = pred_test_y[:, 1]
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        cv_scores.append(log_loss(val_y, pred_val_y))
    print('cv scores : ', cv_scores)
    print('Mean cv score : ', np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    print('Confusion matrix : {}'.format(confusion_matrix(train_y, pred_train > 0.5).ravel()))
    train['word_tfidf_lr_' + target] = pred_train
    test['word_tfidf_lr_' + target] = pred_full_test


print_step('SVD Word 1/4')
n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
print_step('SVD Word 2/4')
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
print_step('SVD Word 3/4')
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
print_step('SVD Word 4/4')
    
train_svd.columns = ['svd_word_' + str(i) for i in range(n_comp)]
test_svd.columns = ['svd_word_' + str(i) for i in range(n_comp)]
train = pd.concat([train, train_svd], axis=1)
test = pd.concat([test, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd


print_step('TFIDF Char 1/3')
tfidf_vec = TfidfVectorizer(ngram_range=(3, 6),
                            analyzer='char',
                            min_df=MIN_DF,
                            max_features=MAX_FEATURES)
full_tfidf = tfidf_vec.fit_transform(train['comment_text'].values.tolist() + test['comment_text'].values.tolist())
print_step('TFIDF Char 2/3')
train_tfidf = tfidf_vec.transform(train['comment_text'].values.tolist())
print_step('TFIDF Char 3/3')
test_tfidf = tfidf_vec.transform(test['comment_text'].values.tolist())
print('TFIDF Char train shape: {}'.format(train_tfidf.shape))
print('TFIDF Char test shape: {}'.format(test_tfidf.shape))


print_step('MNB TFIDF Char')
for target in targets:
    print(target)
    train_y = train[target]
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros(train.shape[0])
    for dev_index, val_index in kf.split(train_X, train_y):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_val_y = pred_val_y[:, 1]
        pred_test_y = pred_test_y[:, 1]
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        cv_scores.append(log_loss(val_y, pred_val_y))
    print('cv scores : ', cv_scores)
    print('Mean cv score : ', np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    print('Confusion matrix : {}'.format(confusion_matrix(train_y, pred_train > 0.5).ravel()))
    train['char_tfidf_mnb_' + target] = pred_train
    test['char_tfidf_mnb_' + target] = pred_full_test


print_step('LR TFIDF Char')
for target in targets:
    print(target)
    train_y = train[target]
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros(train.shape[0])
    for dev_index, val_index in kf.split(train_X, train_y):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runLR(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_val_y = pred_val_y[:, 1]
        pred_test_y = pred_test_y[:, 1]
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        cv_scores.append(log_loss(val_y, pred_val_y))
    print('cv scores : ', cv_scores)
    print('Mean cv score : ', np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    print('Confusion matrix : {}'.format(confusion_matrix(train_y, pred_train > 0.5).ravel()))
    train['char_tfidf_lr_' + target] = pred_train
    test['char_tfidf_lr_' + target] = pred_full_test


print_step('SVD Char 1/4')
n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
print_step('SVD Char 2/4')
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
print_step('SVD Char 3/4')
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
print_step('SVD Char 4/4')
    
train_svd.columns = ['svd_char_' + str(i) for i in range(n_comp)]
test_svd.columns = ['svd_char_' + str(i) for i in range(n_comp)]
train = pd.concat([train, train_svd], axis=1)
test = pd.concat([test, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd


print('Train shape: {}'.format(train.shape))
print('Test shape: {}'.format(test.shape))


print_step('Cache Level 2')
train.to_csv('cache/train_lvl1.csv', index=False)
test.to_csv('cache/test_lvl1.csv', index=False)


print_step('Redrop')
train_X = train.drop(cols_to_drop + targets, axis=1)
test_X = test.drop(cols_to_drop, axis=1)
print('Train shape: {}'.format(train.shape))
print('Test shape: {}'.format(test.shape))


print_step('Level 2 XGB')
for target in targets:
    print(target)
    train_y = train[target]
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros(train.shape[0])
    for dev_index, val_index in kf.split(train_X, train_y):
        dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0)
        pred_val_y = pred_val_y[:, 1]
        pred_test_y = pred_test_y[:, 1]
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        cv_scores.append(log_loss(val_y, pred_val_y))
    print(target)
    print('cv scores : ', cv_scores)
    print('Mean cv score : ', np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    print('Confusion matrix : {}'.format(confusion_matrix(train_y, pred_train > 0.5).ravel()))
    train['lvl2_xgb_' + target] = pred_train
    test['lvl2_xgb_' + target] = pred_full_test


print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['toxic'] = test['lvl2_xgb_toxic']
submission['severe_toxic'] = test['lvl2_xgb_severe_toxic']
submission['obscene'] = test['lvl2_xgb_obscene']
submission['threat'] = test['lvl2_xgb_threat']
submission['insult'] = test['lvl2_xgb_insult']
submission['identity_hate'] = test['lvl2_xgb_identity_hate']
submission.to_csv('submit/submit_lvl2_xgb.csv')
print_step('Done')
