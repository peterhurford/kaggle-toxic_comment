import re
import os

import pandas as pd

from nltk.corpus import stopwords

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


stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u' '.join(
        [x for x in [y for y in non_alphanums.sub(" ", text).lower().strip().split(' ')] if len(x) > 1])


# https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram/code
def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # Go to bytes_strings as I had issues removing all \n in r""
    clean = bytes(text.lower(), encoding='utf-8')
    # 2. Drop \n and  \t
    clean = clean.replace(b'\n', b' ')
    clean = clean.replace(b'\t', b' ')
    clean = clean.replace(b'\b', b' ')
    clean = clean.replace(b'\r', b' ')
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    # 4. Drop puntuation
    # I could have used regex package with regex.sub(b"\p{P}", " ")
    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    clean = b' '.join([exclude.sub(b'', token) for token in clean.split()])
    # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
    clean = re.sub(b'\d+', b' ', clean)
    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
    clean = re.sub(b'\s+', b' ', clean)
    # Remove ending space if any
    clean = re.sub(b'\s+$', b'', clean)
    # 7. Now replace words by words surrounded by # signs
    # e.g. my name is bond would become #my# #name# #is# #bond#
    # clean = re.sub(b'([a-z]+)', b'#\g<1>#', clean)
    clean = re.sub(b' ', b'# #', clean)  # Replace space
    clean = b'#' + clean + b'#'  # add leading and trailing #
    return str(clean, 'utf-8')


def char_analyzer(text):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]
