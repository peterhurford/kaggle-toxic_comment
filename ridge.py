# Adapted from https://www.kaggle.com/rednivrug/5-fold-ridge-oof/code
import re
import gc
import numpy as np
import pandas as pd

import pathos.multiprocessing as mp

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import Ridge

from cv import run_cv_model
from utils import print_step
from preprocess import run_tfidf, normalize_text, clean_text
from cache import get_data, is_in_cache, load_cache, save_in_cache


# Ridge Model Definition
def runRidge(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    model = Ridge(alpha=20, copy_X=True, fit_intercept=True, solver='auto',
            max_iter=100, normalize=False, random_state=42, tol=0.0025)
    model.fit(train_X, train_y)
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # 2. Drop \n and  \t
    # 3. Replace english contractions
    # 4. Drop puntuation
    # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
    clean = normalize_text(text)
    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
    clean = re.sub('\s+', ' ', clean)
    # Remove ending space if any
    clean = re.sub('\s+$', '', clean)
    # 7. Now replace words by words surrounded by # signs
    # e.g. my name is bond would become #my# #name# #is# #bond#
    # clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)
    clean = re.sub(" ", "# #", clean)  # Replace space
    clean = "#" + clean + "#"  # add leading and trailing #
    return str(clean)


def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))


def get_indicators_and_clean_comments(df):
    """
    Check all sorts of content as it may help find toxic comment
    Though I'm not sure all of them improve scores
    """
    # Count number of \n
    df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))
    # Get length in words and characters
    df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))
    # Check number of upper case, if you're angry you may write in upper case
    df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))
    # Number of F words - f..k contains folk, fork,
    df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
    # Number of S word
    df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    # Number of D words
    df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    # Number of occurence of You, insulting someone usually needs someone called : you
    df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    # Just to check you really refered to my mother ;-)
    df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    # Just checking for toxic 19th century vocabulary
    df["nb_ng"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
    # Some Sentences start with a <:> so it may help
    df["start_with_columns"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"^\:+", x))
    # Check for time stamp
    df["has_timestamp"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
    # Check for dates 18:44, 8 December 2010
    df["has_date_long"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
    # Check for date short 8 December 2010
    df["has_date_short"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
    # Check for http links
    df["has_http"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
    # check for mail
    df["has_mail"] = df["comment_text"].apply(
        lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
    )
    # Looking for words surrounded by == word == or """" word """"
    df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
    df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))

    df["chick_count"] = df["comment_text"].apply(lambda x: x.count("!"))
    df["qmark_count"] = df["comment_text"].apply(lambda x: x.count("?"))

    # Now clean comments
    df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))

    # Get the new length in words and characters
    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))
    # Number of different characters used in a comment
    # Using the f word only will reduce the number of letters required in the comment
    df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
    df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(
        lambda x: 1 + min(99, len(x)))


def char_analyzer(text):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]

def clean_csr(csr_trn, csr_sub, min_df):
    trn_min = np.where(csr_trn.getnnz(axis=0) >= min_df)[0]
    sub_min = {x for x in np.where(csr_sub.getnnz(axis=0) >= min_df)[0]}
    mask= [x for x in trn_min if x in sub_min]
    return csr_trn[:, mask], csr_sub[:, mask]

def get_numerical_features(trn, sub):
    """
    As @bangda suggested FM_FTRL either needs to scaled output or dummies
    So here we go for dummies
    """
    ohe = OneHotEncoder()
    full_csr = ohe.fit_transform(np.vstack((trn.values, sub.values)))
    csr_trn = full_csr[:trn.shape[0]]
    csr_sub = full_csr[trn.shape[0]:]
    del full_csr
    gc.collect()
    # Now remove features that don't have enough samples either in train or test
    return clean_csr(csr_trn, csr_sub, 3)


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = get_data()


if not is_in_cache('fm_data'):
    print('~~~~~~~~~~~~~')
    print_step('Cleaning 1/2')
    train_cleaned = get_indicators_and_clean_comments(train)
    print_step('Cleaning 2/2')
    test_cleaned = get_indicators_and_clean_comments(test)
    train_text = train['clean_comment'].fillna('')
    test_text = test['clean_comment'].fillna('')
    all_text = pd.concat([train_text, test_text])

    class_names = ['toxic', 'severe_toxic', 'insult', 'threat', 'obscene', 'identity_hate']
    num_features = [f_ for f_ in train.columns
                    if f_ not in ['comment_text', 'clean_comment', 'id', 'remaining_chars',
                                  'has_ip_address'] + class_names]

    # FM_FTRL likes categorical data
    print_step('Get Numerics 1/2')
    for f in num_features:
        all_cut = pd.cut(pd.concat([train[f], test[f]], axis=0), bins=20, labels=False, retbins=False)
        train[f] = all_cut.values[:train.shape[0]]
        test[f] = all_cut.values[train.shape[0]:]

    print_step('Get Numerics 2/2')
    train_num_features, test_num_features = get_numerical_features(train[num_features], test[num_features])

    print_step('Vectorizer 1/3')
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        tokenizer=lambda x: re.findall(r'[^\p{P}\W]+', x),
        analyzer='word',
        token_pattern=None,
        stop_words='english',
        ngram_range=(1, 2), 
        max_features=300000)
    X = word_vectorizer.fit_transform(all_text)
    train_word_features = X[:train.shape[0]]
    test_word_features = X[train.shape[0]:]
    del (X)

    print_step('Vectorizer 2/3')
    subword_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        tokenizer=char_analyzer,
        analyzer='word',
        ngram_range=(1, 3),
        max_features=60000)
    X = subword_vectorizer.fit_transform(all_text)
    train_subword_features = X[:train.shape[0]]
    test_subword_features = X[train.shape[0]:]
    del (X)
        
    print_step('Vectorizer 3/3')
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        ngram_range=(1, 3),
        max_features=60000)
    X = char_vectorizer.fit_transform(all_text)
    train_char_features = X[:train.shape[0]]
    test_char_features = X[train.shape[0]:]
    del (X)

    print_step('Merging 1/2')
    train_features = hstack([
            train_char_features,
            train_word_features,
            train_num_features,
            train_subword_features
        ]).tocsr()
    del train_word_features, train_num_features, train_char_features,  train_subword_features
    gc.collect()

    print_step('Merging 2/2')
    test_features = hstack([
            test_char_features,
            test_word_features,
            test_num_features,
            test_subword_features
        ]).tocsr()
    del test_word_features, test_num_features, test_char_features, test_subword_features
    gc.collect()

    print("Shapes just to be sure : ", train_features.shape, test_features.shape)
    print_step('Saving')
    save_in_cache('fm_data', train_features, test_features)
    del train_features
    del test_features
    gc.collect()


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~')
print_step('Run Ridge')
train, test = run_cv_model(label='ridge',
                           data_key='fm_data',
                           model_fn=runRidge,
                           train=train,
                           test=test,
                           kf=kf)
import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 1')
save_in_cache('lvl1_ridge', train, test)
# toxic CV scores : [0.9809843104555062, 0.9818160662139189, 0.9810818473334081, 0.9785535369240607, 0.9805031449391929]
# toxic mean CV : 0.9805877811732173
# severe_toxic CV scores : [0.9910152906145414, 0.989781576288062, 0.9905538900693087, 0.9910741898469113, 0.9895167135389562]
# severe_toxic mean CV : 0.990388332071556
# obscene CV scores : [0.9928806730585695, 0.99347239882342, 0.9933801187817354, 0.9926410905084246, 0.9931233899573142]
# obscene mean CV : 0.9930995342258928
# threat CV scores : [0.9898491598311281, 0.9926748758603351, 0.9905821469352692, 0.9904258099519968, 0.9854784282977856]
# threat mean CV : 0.9898020841753029
# insult CV scores : [0.982278876027455, 0.9845963748391066, 0.9861648347221372, 0.9873515099481677, 0.9855485445985808]
# insult mean CV : 0.9851880280270894
# identity_hate CV scores : [0.9809006571379009, 0.9864095257070272, 0.9826035314038123, 0.9880836411995086, 0.9857649558048586]
# identity_hate mean CV : 0.9847524622506215
# ('fm overall : ', 0.9873030369872798)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['toxic'] = test['ridge_toxic']
submission['severe_toxic'] = test['ridge_severe_toxic']
submission['obscene'] = test['ridge_obscene']
submission['threat'] = test['ridge_threat']
submission['insult'] = test['ridge_insult']
submission['identity_hate'] = test['ridge_identity_hate']
submission.to_csv('submit/submit_lvl1_ridge.csv', index=False)
print_step('Done')
