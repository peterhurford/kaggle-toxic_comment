# Neptune OOF models from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/51836
import pandas as pd
from sklearn.metrics import roc_auc_score
from cache import get_data, save_in_cache

base = '~/Downloads/single_model_predictions_03092018/'
train_tail = '_predictions_train_oof.csv'
test_tail = '_predictions_test_oof.csv'
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
all_models = ['count_logreg', 'bad_word_logreg', 'tfidf_logreg', 'char_vdcnn', 'glove_gru',
              'glove_lstm', 'glove_scnn', 'fasttext_dpcnn', 'fasttext_gru', 'fasttext_scnn',
              'glove_dpcnn', 'word2vec_scnn', 'fasttext_lstm', 'word2vec_gru', 'word2vec_lstm',
              'word2vec_dpcnn']

train, test = get_data()
train.drop(['comment_text'], axis=1, inplace=True)
test.drop(['comment_text'], axis=1, inplace=True)
for model in all_models:
    train_ = pd.read_csv(base + model + train_tail).drop(['fold_id'], axis=1)
    test_ = (pd.read_csv(base + model + test_tail)
             .groupby('id')
             .mean()
             .drop(['fold_id'], axis=1)
             .reset_index())
    for label in labels:
        train_['neptune_' + model + '_' + label] = train_[label]
        train_.drop(label, axis=1, inplace=True)
        test_['neptune_' + model + '_' + label] = test_[label]
        test_.drop(label, axis=1, inplace=True)
    train = pd.merge(train, train_, on='id')
    test = pd.merge(test, test_, on='id')
    for label in labels:
        print(model + ' ' + label + ' AUC: ' + str(roc_auc_score(train[label], train['neptune_' + model + '_' + label])))

print('Saving...')
save_in_cache('neptune_models', train, test)
print('Done')
