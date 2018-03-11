import re
import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from utils import print_step


def clean_text(train, test):
    train2 = train.copy()
    test2 = test.copy()
    repl = {
        "&lt;3": " good ",
        ":d": " good ",
        ":dd": " good ",
        ":p": " good ",
        "8)": " good ",
        ":-)": " good ",
        ":)": " good ",
        ";)": " good ",
        "(-:": " good ",
        "(:": " good ",
        "yay!": " good ",
        "yay": " good ",
        "yaay": " good ",
        "yaaay": " good ",
        "yaaaay": " good ",
        "yaaaaay": " good ",
        ":/": " bad ",
        ":&gt;": " sad ",
        ":')": " sad ",
        ":-(": " bad ",
        ":(": " bad ",
        ":s": " bad ",
        ":-s": " bad ",
        "&lt;3": " heart ",
        ":d": " smile ",
        ":p": " smile ",
        ":dd": " smile ",
        "8)": " smile ",
        ":-)": " smile ",
        ":)": " smile ",
        ";)": " smile ",
        "(-:": " smile ",
        "(:": " smile ",
        ":/": " worry ",
        ":&gt;": " angry ",
        ":')": " sad ",
        ":-(": " sad ",
        ":(": " sad ",
        ":s": " sad ",
        ":-s": " sad ",
        r"\br\b": "are",
        r"\bu\b": "you",
        r"\bhaha\b": "ha",
        r"\bhahaha\b": "ha",
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bhasn't\b": "has not",
        r"\bhaven't\b": "have not",
        r"\bhadn't\b": "had not",
        r"\bwon't\b": "will not",
        r"\bwouldn't\b": "would not",
        r"\bcan't\b": "can not",
        r"\bcannot\b": "can not",
        r"\bi'm\b": "i am",
        "m": "am",
        "r": "are",
        "u": "you",
        "haha": "ha",
        "hahaha": "ha",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "won't": "will not",
        "wouldn't": "would not",
        "can't": "can not",
        "cannot": "can not",
        "i'm": "i am",
        "m": "am",
        "i'll" : "i will",
        "its" : "it is",
        "it's" : "it is",
        "'s" : " is",
        "that's" : "that is",
        "weren't" : "were not",
    }

    keys = [i for i in repl.keys()]

    new_train_data = []
    new_test_data = []
    ltr = train['comment_text'].tolist()
    lte = test['comment_text'].tolist()
    for i in ltr:
        arr = str(i).split()
        xx = ""
        for j in arr:
            j = str(j).lower()
            if j[:4] == 'http' or j[:3] == 'www':
                continue
            if j in keys:
                # print("inn")
                j = repl[j]
            xx += j + ' '
        new_train_data.append(xx)
    for i in lte:
        arr = str(i).split()
        xx = ''
        for j in arr:
            j = str(j).lower()
            if j[:4] == 'http' or j[:3] == 'www':
                continue
            if j in keys:
                # print("inn")
                j = repl[j]
            xx += j + ' '
        new_test_data.append(xx)
    train2['comment_text'] = new_train_data
    test2['comment_text'] = new_test_data
    print_step('crap removed')
    trate = train2['comment_text'].tolist()
    tete = test2['comment_text'].tolist()
    for i, c in enumerate(trate):
        trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
    for i, c in enumerate(tete):
        tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
    train2['comment_text'] = trate
    test2['comment_text'] = tete
    print_step('only alphabets')

    train2['comment_text'].fillna('missing', inplace=True)
    test2['comment_text'].fillna('missing', inplace=True)
    print_step('Filling missing')
    print('Train shape: {}'.format(train.shape))
    print('Test shape: {}'.format(test.shape))
    return train2, test2


def run_tfidf(train, test, ngram_min=1, ngram_max=2, min_df=5,
              max_features=20000, rm_stopwords=True, analyzer='word',
              sublinear_tf=False, token_pattern=r'(?u)\b\w\w+\b', binary=False,
              tokenize=False, tokenizer=None):
    rm_stopwords = 'english' if rm_stopwords else None
    strip_accents = 'unicode' if tokenize else None
    tfidf_vec = TfidfVectorizer(ngram_range=(ngram_min, ngram_max),
                                analyzer=analyzer,
                                stop_words=rm_stopwords,
                                strip_accents=strip_accents,
                                token_pattern=token_pattern,
                                tokenizer=tokenizer,
                                min_df=min_df,
                                max_features=max_features,
                                sublinear_tf=sublinear_tf,
                                binary=binary)
    print_step('TFIDF ngrams ' + str(ngram_min) + ' to ' + str(ngram_max) + ' on ' +
               str(analyzer) + ' with strip accents = ' + str(strip_accents) +
               ', token_pattern = ' + str(token_pattern) + ', tokenizer = ' + str(tokenizer) +
               ', rm_stopwords = ' + str(rm_stopwords) + ', min_df = ' + str(min_df) +
               ', max_features = ' + str(max_features) + ', sublinear_tf = ' +
               str(sublinear_tf) + ', binary = ' + str(binary))
    train_tfidf = tfidf_vec.fit_transform(train['comment_text'])
    print_step('TFIDF 1/2')
    test_tfidf = tfidf_vec.transform(test['comment_text'])
    print_step('TFIDF 2/2')
    print('TFIDF train shape: {}'.format(train_tfidf.shape))
    print('TFIDF test shape: {}'.format(test_tfidf.shape))
    return train_tfidf, test_tfidf


non_alphas = re.compile(u'[^A-Za-z]+')
cont_patterns = [
    ('(W|w)on\'t', 'will not'),
    ('(C|c)an\'t', 'can not'),
    ('(I|i)\'m', 'i am'),
    ('(A|a)in\'t', 'is not'),
    ('(\w+)\'ll', '\g<1> will'),
    ('(\w+)n\'t', '\g<1> not'),
    ('(\w+)\'ve', '\g<1> have'),
    ('(\w+)\'s', '\g<1> is'),
    ('(\w+)\'re', '\g<1> are'),
    ('(\w+)\'d', '\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]
def normalize_text(text):
    clean = text.lower()
    clean = clean.replace('\n', ' ')
    clean = clean.replace('\t', ' ')
    clean = clean.replace('\b', ' ')
    clean = clean.replace('\r', ' ')
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    return u' '.join([y for y in non_alphas.sub(' ', clean).strip().split(' ')])
