# Following https://www.kaggle.com/eashish/bidirectional-gru-with-convolution/notebook
from pprint import pprint

import pandas as pd
import numpy as np
np.random.seed(42)

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback
from keras.optimizers import Adam

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from utils import print_step
from cache import get_data, is_in_cache, save_in_cache
from preprocess import normalize_text, glove_preprocess


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
epochs = 4
batch_size = 128
predict_batch_size = 1024


if not is_in_cache('lvl1_gru-conv'):
    print_step('Loading data')
    train_df, test_df = get_data()
    print_step('Preprocessing 1/3')
    train_df['comment_text'] = train_df['comment_text'].apply(glove_preprocess).apply(normalize_text)
    print_step('Preprocessing 2/3')
    test_df['comment_text'] = test_df['comment_text'].apply(glove_preprocess).apply(normalize_text)

    print_step('Preprocessing 3/3')
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
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(x)
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    model.save_weights('cache/gru-conv-model-weights.h5')


    print_step('Making KFold for CV')
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

    i = 1
    cv_scores = []
    pred_train = np.zeros((train_df.shape[0], 6))
    pred_full_test = np.zeros((test_df.shape[0], 6))
    for dev_index, val_index in kf.split(x_train, y_train[:, 0]):
        print_step('Started fold ' + str(i))
        model.load_weights('cache/gru-conv-model-weights.h5')
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
        train_df['gru-conv_' + classx] = pred_train[:, k]
        test_df['gru-conv_' + classx] = pred_full_test[:, k]

    print('~~~~~~~~~~~~~~~~~~')
    print_step('Cache Level 1')
    save_in_cache('lvl1_gru-conv', train_df, test_df)
    print_step('Done!')


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['toxic'] = test_df['gru-conv_toxic']
submission['severe_toxic'] = test_df['gru-conv_severe_toxic']
submission['obscene'] = test_df['gru-conv_obscene']
submission['threat'] = test_df['gru-conv_threat']
submission['insult'] = test_df['gru-conv_insult']
submission['identity_hate'] = test_df['gru-conv_identity_hate']
submission.to_csv('submit/submit_lvl1_gru-conv.csv', index=False)
print_step('Done')
# [('toxic', 0.9809322867932334),
#  ('severe_toxic', 0.9893592324003722),
#  ('obscene', 0.9911515049028361),
#  ('threat', 0.9852941719302706),
#  ('insult', 0.9860467711310973),
#  ('identity_hate', 0.9845829254690504)]
# mean cv score : 0.98622781543781
