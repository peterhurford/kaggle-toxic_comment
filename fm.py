import gc
import numpy as np

from scipy.sparse import csr_matrix, hstack

import wordbatch
from wordbatch.extractors import WordBag

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from wordbatch.models import FM_FTRL

from utils import run_cv_model, print_step
from preprocess import get_data, normalize_text


# Wordbatch
wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 21,
                                                              "norm": 'l2',
                                                              "tf": 1.0,
                                                              "idf": None,
                                                              }), procs=8)
# FM Model Definition
def runFM(train_X, train_y, test_X, test_y, test_X2, label):
    rounds = 10
    model = FM_FTRL(D=train_X.shape[1], D_fm=20, iters=1, inv_link="sigmoid", threads=4)
    for i in range(rounds):
        model.fit(train_X, train_y)
        predsFM = model.predict(test_X)
        print_step('Iteration {}/{} -- AUC: {}'.format(i + 1, rounds, roc_auc_score(test_y, predsFM)))
    predsFM2 = model.predict(test_X2)
    return predsFM, predsFM2, model


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = get_data()

print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

print('~~~~~~~~~~~~~~~~~~')
print_step('Wordbatch 1/3')
wb.dictionary_freeze = True
X_train = wb.fit_transform(train['comment_text'])
print_step('Wordbatch 2/3')
X_test = wb.transform(test['comment_text'])
print_step('Wordbatch 3/3')
del(wb)
mask = np.where(X_train.getnnz(axis=0) > 8)[0]
X_train = X_train[:, mask]
X_test = X_test[:, mask]
print('Wordbatch train shape', X_train.shape)
print('Wordbatch test shape', X_test.shape)

print('~~~~~~~~~~~')
print_step('Run FM')
train, test = run_cv_model(label='word_fm',
                           train=train,
                           test=test,
                           post_train=X_train,
                           post_test=X_test,
                           model_fn=runFM,
                           kf=kf)
# Toxic:   0.97071085751838204
# Severe:  0.9858469322003458
# Obscene: 0.9834975327437192
# Threat:  0.98732987354980428
# Insult:  0.97758018514263978
# Hate:    0.97315936110261991
# 0.97968745704291849

import pdb
pdb.set_trace()
