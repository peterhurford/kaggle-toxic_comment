import gc
from pprint import pprint

import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback

from sklearn.preprocessing import MinMaxScaler, minmax_scale
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from utils import print_step
from cache import get_data, load_cache, save_in_cache


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


epochs = 3
batch_size = 32
predict_batch_size = 1024

print_step('Loading 1/2...')
train_df, test_df = get_data()
print_step('Loading 2/2...')
train_, test_ = load_cache('lvl2_all')
print_step('Scaling 1/2...')
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_)
print_step('Scaling 2/2...')
test_scaled = scaler.transform(test_)
train_ = pd.DataFrame(train_scaled, columns = train_.columns)
test_ = pd.DataFrame(test_scaled, columns = test_.columns)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = train_df[labels].values


pred_train_all = np.zeros((train_df.shape[0], 6))
pred_full_test_all = np.zeros((test_df.shape[0], 6))
for seedx in range(40, 60):
    print('## START BAG ' + str(seedx))
    np.random.seed(seedx)

    print('Bag input...')
    train_bag = train_.sample(frac=0.8, axis=1).reset_index().drop('index', axis=1)
    test_bag = test_[train_bag.columns]
    train_bag = train_bag.values
    test_bag = test_bag.values
    print(train_.shape)
    print(test_.shape)
    print(train_bag.shape)
    print(test_bag.shape)

    print_step('Build model...')
    inp = Input(shape=(train_bag.shape[1], ))
    x = Dense(64, activation='relu')(inp)
    outp = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.save_weights('cache/final-cnn-model-weights.h5')


    print_step('Making KFold for CV')
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seedx)

    i = 1
    cv_scores = []
    pred_train = np.zeros((train_bag.shape[0], 6))
    pred_full_test = np.zeros((test_bag.shape[0], 6))
    for dev_index, val_index in kf.split(train_bag, y_train[:, 0]):
        print_step('Started fold ' + str(i))
        model.load_weights('cache/final-cnn-model-weights.h5')
        dev_X, val_X = train_bag[dev_index], train_bag[val_index]
        dev_y, val_y = y_train[dev_index, :], y_train[val_index, :]
        RocAuc = RocAucEvaluation(validation_data=(val_X, val_y), interval=1)
        hist = model.fit(dev_X, dev_y, batch_size=batch_size, epochs=epochs,
                         validation_data=(val_X, val_y), callbacks=[RocAuc])
        val_pred = model.predict(val_X, batch_size=predict_batch_size, verbose=1)
        test_pred = model.predict(test_bag, batch_size=predict_batch_size, verbose=1)
        for j in range(val_pred.shape[1]):
            val_pred[:, j] = minmax_scale(pd.Series(val_pred[:, j]).rank().values) # Rank transform
            test_pred[:, j] = minmax_scale(pd.Series(test_pred[:, j]).rank().values)
        pred_train[val_index, :] = val_pred
        pred_full_test = pred_full_test + test_pred
        cv_score = [roc_auc_score(val_y[:, j], val_pred[:, j]) for j in range(6)]
        print_step('Fold ' + str(i) + ' done')
        pprint(zip(labels, cv_score))
        cv_scores.append(cv_score)
        i += 1
    print_step('All folds done!')
    print('CV scores')
    pprint(zip(labels, np.mean(cv_scores, axis=0)))
    mean_cv_score = np.mean(np.mean(cv_scores, axis=0))
    print('mean cv score : ' + str(mean_cv_score))
    pred_full_test = pred_full_test / 5.
    pred_train_all += pred_train 
    pred_full_test_all += pred_full_test
    del model
    gc.collect()

aucs = []
for k, label in enumerate(labels):
    train_df['final-cnn_' + label] = pred_train_all[:, k]
    auc = roc_auc_score(train_df[label], pred_train_all[:, k])
    aucs.append(auc)
    print(label + ' AUC: ' + str(auc))
    test_df['final-cnn_' + label] = pred_full_test_all[:, k]
print('Total Mean AUC: ' + str(np.mean(aucs)))

import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 2')
save_in_cache('lvl2_final-cnn', train_df, test_df)
print_step('Done!')


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['toxic'] = test_df['final-cnn_toxic']
submission['severe_toxic'] = test_df['final-cnn_severe_toxic']
submission['obscene'] = test_df['final-cnn_obscene']
submission['threat'] = test_df['final-cnn_threat']
submission['insult'] = test_df['final-cnn_insult']
submission['identity_hate'] = test_df['final-cnn_identity_hate']
submission.to_csv('submit/submit_lvl2_final-cnn.csv', index=False)
print_step('Done')
# toxic AUC: 0.9876654371135581
# severe_toxic AUC: 0.9919113105232602
# obscene AUC: 0.9953169552409985
# threat AUC: 0.9907555452881471
# insult AUC: 0.9897613209151592
# identity_hate AUC: 0.9898029652435525
# Total Mean AUC: 0.990868922387446
