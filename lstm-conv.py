# Following https://www.kaggle.com/fizzbuzz/bi-lstm-conv-layer-lb-score-0-9840/
from pprint import pprint

import pandas as pd
import numpy as np
np.random.seed(42)

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, LSTM, Conv1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, Bidirectional
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from utils import print_step
from cache import is_in_cache, save_in_cache


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


EMBEDDING_FILE = 'cache/glove/glove.840B.300d.txt'


max_features = 100000
maxlen = 150
embed_size = 300
epochs = 1
batch_size = 32
predict_batch_size = 32


if not is_in_cache('lvl1_lstm-conv'):
    print_step('Loading data')
    train_df = pd.read_csv('data/train_zafar_cleaned.csv')
    test_df = pd.read_csv('data/test_zafar_cleaned.csv')
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    X_train = train_df['comment_text'].fillna('peterhurford').values
    y_train = train_df[classes].values
    X_test = test_df['comment_text'].fillna('peterhurford').values

    print_step('Tokenizing data...')
    tokenizer = Tokenizer(num_words=max_features, lower=True)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    x_train = tokenizer.texts_to_sequences(X_train)
    x_test = tokenizer.texts_to_sequences(X_test)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

    print_step('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)


    print_step('Embedding')
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector


    print_step('Build model...')
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.35)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.save_weights('cache/lstm-conv-model-weights.h5')


    print_step('Making KFold for CV')
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

    i = 1
    cv_scores = []
    pred_train = np.zeros((train_df.shape[0], 6))
    pred_full_test = np.zeros((test_df.shape[0], 6))
    for dev_index, val_index in kf.split(x_train, y_train[:, 0]):
        print_step('Started fold ' + str(i))
        model.load_weights('cache/lstm-conv-model-weights.h5')
        dev_X, val_X = x_train[dev_index], x_train[val_index]
        dev_y, val_y = y_train[dev_index, :], y_train[val_index, :]
        RocAuc = RocAucEvaluation(validation_data=(val_X, val_y), interval=1)
        hist = model.fit(dev_X, dev_y, batch_size=batch_size, epochs=epochs,
                         validation_data=(val_X, val_y), callbacks=[RocAuc])
        val_pred = model.predict(val_X, batch_size=predict_batch_size, verbose=1)
        pred_train[val_index, :] = val_pred
        test_pred = model.predict(x_test, batch_size=predict_batch_size, verbose=1)
        pred_full_test = pred_full_test + test_pred
        cv_score = [roc_auc_score(val_y[:, j], val_pred[:, j]) for j in range(6)]
        print_step('Fold ' + str(i) + ' done')
        pprint(zip(classes, cv_score))
        cv_scores.append(cv_score)
        i += 1
    print_step('All folds done!')
    print('CV scores')
    pprint(zip(classes, np.mean(cv_scores, axis=0)))
    mean_cv_score = np.mean(np.mean(cv_scores, axis=0))
    print('mean cv score : ' + str(mean_cv_score))
    pred_full_test = pred_full_test / 5.
    for k, classx in enumerate(classes):
        train_df['lstm-conv_' + classx] = pred_train[:, k]
        test_df['lstm-conv_' + classx] = pred_full_test[:, k]

    print('~~~~~~~~~~~~~~~~~~')
    print_step('Cache Level 1')
    save_in_cache('lvl1_lstm-conv', train_df, test_df)
    print_step('Done!')


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['toxic'] = test_df['lstm-conv_toxic']
submission['severe_toxic'] = test_df['lstm-conv_severe_toxic']
submission['obscene'] = test_df['lstm-conv_obscene']
submission['threat'] = test_df['lstm-conv_threat']
submission['insult'] = test_df['lstm-conv_insult']
submission['identity_hate'] = test_df['lstm-conv_identity_hate']
submission.to_csv('submit/submit_lvl1_lstm-conv.csv', index=False)
print_step('Done')
# CV scores
# [('toxic', 0.9832108709063038),
#  ('severe_toxic', 0.9905899996144854),
#  ('obscene', 0.9930442541213852),
#  ('threat', 0.9887570976973616),
#  ('insult', 0.9877404846287681),
#  ('identity_hate', 0.9884367797351654)]
# mean cv score : 0.9886299144505782
