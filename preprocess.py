import re

import pandas as pd

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from utils import print_step


def get_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    print('Train shape: {}'.format(train.shape))
    print('Test shape: {}'.format(test.shape))


    print_step('Filling missing')
    train['comment_text'].fillna('missing', inplace=True)
    test['comment_text'].fillna('missing', inplace=True)
    print('Train shape: {}'.format(train.shape))
    print('Test shape: {}'.format(test.shape))
    return train, test


def get_stage2_data():
    train = pd.read_csv('cache/train_lvl1.csv')
    test = pd.read_csv('cache/test_lvl1.csv')
    print('Train shape: {}'.format(train.shape))
    print('Test shape: {}'.format(test.shape))
    return train, test


def run_tfidf(train, test, ngram_min=1, ngram_max=2, min_df=5,
              max_features=20000, rm_stopwords=True, analyzer='word',
              sublinear_tf=False, token_pattern=r'(?u)\b\w\w+\b', binary=False,
              tokenize=False):
    rm_stopwords = 'english' if rm_stopwords else None
    strip_accents = 'unicode' if tokenize else None
    tfidf_vec = TfidfVectorizer(ngram_range=(ngram_min, ngram_max),
                                analyzer=analyzer,
                                stop_words=rm_stopwords,
                                strip_accents=strip_accents,
                                token_pattern=token_pattern,
                                min_df=min_df,
                                max_features=max_features,
                                sublinear_tf=sublinear_tf,
                                binary=binary)
    print_step('TFIDF ngrams ' + str(ngram_min) + ' to ' + str(ngram_max) + ' on ' +
               str(analyzer) + ' with strip accents = ' + str(strip_accents) +
               ', rm_stopwords = ' + str(rm_stopwords) + ', min_df = ' + str(min_df) +
               ', max_features = ' + str(max_features) + ', sublinear_tf = ' +
               str(sublinear_tf) + ', binary = ' + str(binary))
    train_tfidf = tfidf_vec.fit_transform(train['comment_text'])
    print_step('TFIDF 1/2')
    test_tfidf = tfidf_vec.transform(test['comment_text'])
    print_step('TFIDF 2/2')
    print('TFIDF Word train shape: {}'.format(train_tfidf.shape))
    print('TFIDF Word test shape: {}'.format(test_tfidf.shape))
    return train_tfidf, test_tfidf


stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])
