# Following https://www.kaggle.com/yekenot/pooled-attention-lstm-fasttext/code
from pprint import pprint

import pandas as pd
import numpy as np
np.random.seed(42)

from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from utils import print_step
from cache import get_data, is_in_cache, save_in_cache


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


EMBEDDING_FILE = 'cache/crawl/crawl-300d-2M.vec'


max_features = 100000
maxlen = 150
embed_size = 300
epochs = 8
batch_size = 256
predict_batch_size = 1024


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim


if not is_in_cache('lvl1_attention-lstm'):
    train_df, test_df = get_data()

    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    X_train = train_df['comment_text'].fillna('peterhurford').values
    y_train = train_df[classes].values
    X_test = test_df['comment_text'].fillna('peterhurford').values

    print_step('Tokenizing data...')
    tokenizer = Tokenizer(num_words=max_features)
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
    x = LSTM(300, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)(x)
    x = Dropout(0.25)(x)
    x = Attention(150)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    outp = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.save_weights('cache/attention-lstm-model-weights.h5')


    print_step('Making KFold for CV')
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

    i = 1
    cv_scores = []
    pred_train = np.zeros((train_df.shape[0], 6))
    pred_full_test = np.zeros((test_df.shape[0], 6))
    for dev_index, val_index in kf.split(x_train, y_train[:, 0]):
        print_step('Started fold ' + str(i))
        model.load_weights('cache/attention-lstm-model-weights.h5')
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
        train_df['attention-lstm_' + classx] = pred_train[:, k]
        test_df['attention-lstm_' + classx] = pred_full_test[:, k]

    print('~~~~~~~~~~~~~~~~~~')
    print_step('Cache Level 1')
    save_in_cache('lvl1_attention-lstm', train_df, test_df)
    print_step('Done!')


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['toxic'] = test_df['attention-lstm_toxic']
submission['severe_toxic'] = test_df['attention-lstm_severe_toxic']
submission['obscene'] = test_df['attention-lstm_obscene']
submission['threat'] = test_df['attention-lstm_threat']
submission['insult'] = test_df['attention-lstm_insult']
submission['identity_hate'] = test_df['attention-lstm_identity_hate']
submission.to_csv('submit/submit_lvl1_attention-lstm.csv', index=False)
print_step('Done')
# [('toxic', 0.9820368681754601),
#  ('severe_toxic', 0.9869527277812544),
#  ('obscene', 0.9907345344290507),
#  ('threat', 0.9802004162414051),
#  ('insult', 0.9860725232467195),
#  ('identity_hate', 0.9823505498076681)]
# mean cv score : 0.9847246032802598

