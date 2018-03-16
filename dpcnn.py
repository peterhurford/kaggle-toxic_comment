# Following https://www.kaggle.com/michaelsnell/conv1d-dpcnn-in-keras/code
from pprint import pprint

import pandas as pd
import numpy as np
np.random.seed(42)

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, GlobalMaxPooling1D, add
from keras.layers import Conv1D, BatchNormalization, PReLU, MaxPooling1D, Dropout
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback

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
maxlen = 200
embed_size = 300
epochs = 4
batch_size = 128
predict_batch_size = 1024
filter_nr = 64
filter_size = 3
max_pool_size = 3
max_pool_strides = 2
dense_nr = 256
spatial_dropout = 0.2
dense_dropout = 0.5
train_embed = False

def schedule(ind):
    a = [0.001, 0.0005, 0.0001, 0.0001]
    return a[ind] 


if not is_in_cache('lvl1_dpcnn'):
    print_step('Loading data')
    train_df, test_df = get_data()
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
    comment = Input(shape=(maxlen,))
    emb_comment = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=train_embed)(comment)
    emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)

    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(emb_comment)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)

#we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
#if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
    resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(emb_comment)
    resize_emb = PReLU()(resize_emb)
        
    block1_output = add([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1_output)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
        
    block2_output = add([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2_output)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
        
    block3_output = add([block3, block2_output])
    block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3_output)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)
    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block4)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)

    output = add([block4, block3_output])
    output = GlobalMaxPooling1D()(output)
    output = Dense(dense_nr, activation='linear')(output)
    output = BatchNormalization()(output)
    output = PReLU()(output)
    output = Dropout(dense_dropout)(output)
    output = Dense(6, activation='sigmoid')(output)

    model = Model(comment, output)


    model.compile(loss='binary_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])
    model.save_weights('cache/dpcnn-model-weights.h5')


    print_step('Making KFold for CV')
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

    i = 1
    cv_scores = []
    pred_train = np.zeros((train_df.shape[0], 6))
    pred_full_test = np.zeros((test_df.shape[0], 6))
    for dev_index, val_index in kf.split(x_train, y_train[:, 0]):
        print_step('Started fold ' + str(i))
        model.load_weights('cache/dpcnn-model-weights.h5')
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
        train_df['dpcnn_' + classx] = pred_train[:, k]
        test_df['dpcnn_' + classx] = pred_full_test[:, k]

    print('~~~~~~~~~~~~~~~~~~')
    print_step('Cache Level 1')
    save_in_cache('lvl1_dpcnn', train_df, test_df)
    print_step('Done!')


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['toxic'] = test_df['dpcnn_toxic']
submission['severe_toxic'] = test_df['dpcnn_severe_toxic']
submission['obscene'] = test_df['dpcnn_obscene']
submission['threat'] = test_df['dpcnn_threat']
submission['insult'] = test_df['dpcnn_insult']
submission['identity_hate'] = test_df['dpcnn_identity_hate']
submission.to_csv('submit/submit_lvl1_dpcnn.csv', index=False)
print_step('Done')
# [('toxic', 0.9809322867932334),
#  ('severe_toxic', 0.9893592324003722),
#  ('obscene', 0.9911515049028361),
#  ('threat', 0.9852941719302706),
#  ('insult', 0.9860467711310973),
#  ('identity_hate', 0.9845829254690504)]
# mean cv score : 0.98622781543781
