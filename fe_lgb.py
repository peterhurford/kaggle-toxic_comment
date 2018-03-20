import gc

import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

from scipy.sparse import csr_matrix, hstack

from sklearn.decomposition import TruncatedSVD

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step
from preprocess import run_tfidf, normalize_text, glove_preprocess
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


# Embedding params
max_features = 100000
maxlen = 200
embed_size = 200


# LGB Model Definition
def runLGB(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    params = {'learning_rate': 0.05,
              'application': 'binary',
              'num_leaves': 31,
              'verbosity': -1,
              'metric': 'auc',
              'data_random_seed': 3,
              'bagging_fraction': 1.0,
              'feature_fraction': 0.4,
              'nthread': min(mp.cpu_count() - 1, 6),
              'lambda_l1': 1,
              'lambda_l2': 1}
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
    print(model.feature_importance())
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


if is_in_cache('lgb_fe_with_embeddings_and_svd'):
    train, test = load_cache('lgb_fe_with_embeddings_and_svd')
else:
    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Importing Data')
    train, test = get_data()
    if is_in_cache('fe_lgb_data'):
        train_fe, test_fe = load_cache('fe_lgb_data')
    else:
        print_step('Adding Features')
        train_fe, test_fe = add_features(train, test)
        print_step('Dropping')
        train_fe.drop(['id', 'comment_text'], axis=1, inplace=True)
        test_fe.drop(['id', 'comment_text'], axis=1, inplace=True)
        train_fe.drop(['toxic', 'severe_toxic', 'obscene', 'threat',
                       'insult', 'identity_hate'], axis=1, inplace=True)
        print_step('Saving')
        save_in_cache('fe_lgb_data', train_fe, test_fe)


    print('~~~~~~~~~~~~~~~~~~~~~')
    print_step('Adding Embedding')
    EMBEDDING_FILE = 'cache/twitter/glove.twitter.27B.200d.txt'

    print_step('Embedding 1/7')
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

    def text_to_embedding(text):
        mean = np.mean([embeddings_index.get(w, np.zeros(embed_size)) for w in text.split()], axis=0)
        if mean.shape == ():
            return np.zeros(embed_size)
        else:
            return mean

    print_step('Embedding 2/7')
    train_embeddings = (train['comment_text']
                        .fillna('peterhurford')
                        .apply(glove_preprocess)
                        .apply(normalize_text)
                        .apply(text_to_embedding))

    print_step('Embedding 3/7')
    test_embeddings = (test['comment_text']
                       .fillna('peterhurford')
                       .apply(glove_preprocess)
                       .apply(normalize_text)
                       .apply(text_to_embedding))

    print_step('Embedding 4/7')
    train_embeddings_df = pd.DataFrame(train_embeddings.values.tolist(),
                                       columns = ['embed' + str(i) for i in range(embed_size)])
    print_step('Embedding 5/7')
    test_embeddings_df = pd.DataFrame(test_embeddings.values.tolist(),
                                      columns = ['embed' + str(i) for i in range(embed_size)])

    print_step('Embedding 6/7')
    train = pd.concat([train, train_embeddings_df], axis=1)
    print_step('Embedding 7/7')
    test = pd.concat([test, test_embeddings_df], axis=1)


    print('~~~~~~~~~~~~~~~')
    print_step('Adding SVD')
    svd = TruncatedSVD(embed_size, algorithm='arpack', random_state=42)

    print_step('Run TFIDF WORD-CHAR UNION')
    if is_in_cache('tfidf_char_union'):
        post_train, post_test = load_cache('tfidf_char_union')
    else:
        TFIDF_UNION1.update({'train': train, 'test': test})
        post_trainw, post_testw = run_tfidf(**TFIDF_UNION1)
        TFIDF_UNION2.update({'train': train, 'test': test})
        post_trainc, post_testc = run_tfidf(**TFIDF_UNION2)
        post_train = csr_matrix(hstack([post_trainw, post_trainc]))
        del post_trainw; del post_trainc; gc.collect()
        post_test = csr_matrix(hstack([post_testw, post_testc]))
        del post_testw; del post_testc; gc.collect()
        save_in_cache('tfidf_char_union', post_train, post_test)

    print_step('Truncating with SVD 1/4')
    train_svd = svd.fit_transform(post_train)
    print_step('Truncating with SVD 2/4')
    test_svd = svd.transform(post_test)
    print_step('Truncating with SVD 3/4')
    train_svd_df = pd.DataFrame(train_svd, columns = ['svd' + str(i) for i in range(embed_size)])
    test_svd_df = pd.DataFrame(test_svd, columns = ['svd' + str(i) for i in range(embed_size)])
    print_step('Truncating with SVD 4/4')
    train = pd.concat([train, train_svd_df], axis=1)
    test = pd.concat([test, test_svd_df], axis=1)
    print_step('Merging')
    train = pd.concat([train, train_fe], axis=1)
    test = pd.concat([test, test_fe], axis=1)
    train.drop(['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)
    test.drop(['id', 'comment_text'], axis=1, inplace=True)
    print_step('Saving')
    save_in_cache('lgb_fe_with_embeddings_and_svd', train, test)
    del train_svd_df
    del test_svd_df
    del train_svd
    del test_svd
    del train_embeddings_df
    del test_embeddings_df
    del train_embeddings
    del test_embeddings
    del embeddings_index
    del train_fe
    del test_fe
    gc.collect()


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~~')
print_step('Run LGB')
print(train.columns.values)

train, test = get_data()
train, test = run_cv_model(label='fe_lgb',
                           data_key='lgb_fe_with_embeddings_and_svd',
                           model_fn=runLGB,
                           train=train,
                           test=test,
                           kf=kf)

import pdb
pdb.set_trace()
print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('lvl1_fe_lgb', train, test)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['toxic'] = test['fe_lgb_toxic']
submission['severe_toxic'] = test['fe_lgb_severe_toxic']
submission['obscene'] = test['fe_lgb_obscene']
submission['threat'] = test['fe_lgb_threat']
submission['insult'] = test['fe_lgb_insult']
submission['identity_hate'] = test['fe_lgb_identity_hate']
submission.to_csv('submit/submit_fe_lgb.csv', index=False)
print_step('Done!')
# toxic CV scores : [0.9821160871586277, 0.982171088543915, 0.9812966603938744, 0.979891901255327, 0.9818157565754393]
# toxic mean CV : 0.9814582987854367 overall: 0.981440779466493
# severe_toxic CV scores : [0.9909844347584176, 0.9891544682132456, 0.9906661553626646, 0.9923261735890316, 0.9879432135059663]
# severe_toxic mean CV : 0.9902148890858651 overall: 0.9901717065708803
# obscene CV scores : [0.9934223305484997, 0.993378145173526, 0.9928880923816176, 0.9924376294085034, 0.9922531265269842]
# obscene mean CV : 0.9928758648078262 overall: 0.9928651969456183
# threat CV scores : [0.9906453298343757, 0.9929315359271714, 0.9909560063693601, 0.9924121070165514, 0.9874055069788369]
# threat mean CV : 0.9908700972252591 overall: 0.9908104722410858
# insult CV scores : [0.9834516536738911, 0.9842242055634276, 0.9844226274840334, 0.9860190220874488, 0.9866087059343043]
# insult mean CV : 0.984945242948621 overall: 0.9849186078769634
# identity_hate CV scores : [0.9857338504879092, 0.9885726795736648, 0.9823806685054449, 0.9876161466138621, 0.9901515636459199]
# identity_hate mean CV : 0.9868909817653602 overall: 0.9867888564125361
# fe_lgb CV mean : 0.987875895769728, overall: 0.9878326032522629
