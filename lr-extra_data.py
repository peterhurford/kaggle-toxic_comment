import gc
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression, Ridge

from cv import run_cv_model
from utils import print_step
from preprocess import run_tfidf, clean_text
from cache import get_data, is_in_cache, load_cache, save_in_cache

def rmse(actual, predicted):
        return np.sqrt(((predicted - actual) ** 2).mean())


# TFIDF Hyperparams
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


def runSagLR(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    model = LogisticRegression(solver='sag')
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)[:, 1]
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2

def runRidge(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
    model.fit(train_X, train_y)
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~~~~~')
if not is_in_cache('extra_data_attack') and not is_in_cache('extra_data_toxic'):
    print_step('Importing Data 1/5')
    attack = pd.read_csv('data/attack_annotations.tsv', sep='\t')
    print_step('Importing Data 2/5')
    attack_comments = pd.read_csv('data/attack_annotated_comments.tsv', sep='\t')
    print_step('Importing Data 3/5')
    toxic = pd.read_csv('data/toxicity_annotations.tsv', sep='\t')
    print_step('Importing Data 4/5')
    toxic_comments = pd.read_csv('data/toxicity_annotated_comments.tsv', sep='\t')
    print_step('Importing Data 5/5')
    train, test = get_data()

    print_step('Processing 1/9')
    train.drop(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)
    test = pd.concat([train, test])

    print_step('Processing 2/9')
    attack = attack.drop('worker_id', axis=1).groupby('rev_id').mean().reset_index()
    print_step('Processing 3/9')
    attack = attack_comments[['rev_id', 'comment']].merge(attack, on='rev_id').drop('rev_id', axis=1)
    print_step('Processing 4/9')
    attack['attack_score'] = attack['attack']
    attack['quoting_attack'] = attack['quoting_attack'].apply(lambda x: 1 if x > 0.1 else 0)
    attack['recipient_attack'] = attack['recipient_attack'].apply(lambda x: 1 if x > 0.1 else 0)
    attack['third_party_attack'] = attack['third_party_attack'].apply(lambda x: 1 if x > 0.1 else 0)
    attack['other_attack'] = attack['other_attack'].apply(lambda x: 1 if x > 0.1 else 0)
    attack['attack'] = attack['attack_score'].apply(lambda x: 1 if x > 0.1 else 0)
    attack['comment_text'] = attack['comment']
    attack.drop('comment', axis=1, inplace=True)
    print_step('Processing 5/9')
    save_in_cache('extra_data_attack', attack, test)

    print_step('Processing 6/9')
    toxic = toxic.drop('worker_id', axis=1).groupby('rev_id').mean().reset_index()
    print_step('Processing 7/9')
    toxic = toxic_comments[['rev_id', 'comment']].merge(toxic, on='rev_id').drop('rev_id', axis=1)
    print_step('Processing 8/9')
    toxic['toxicity_label'] = toxic['toxicity'].apply(lambda x: 1 if x > 0.1 else 0)
    toxic['comment_text'] = toxic['comment']
    toxic.drop('comment', axis=1, inplace=True)
    print_step('Processing 9/9')
    save_in_cache('extra_data_toxic', toxic, test)
else:
    attack, test = load_cache('extra_data_attack')
    toxic, test = load_cache('extra_data_toxic')


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)
kf_for_regression = KFold(n_splits = 5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run TFIDF WORD-CHAR UNION 1/2')
if is_in_cache('tfidf_char_union_extra_data_attack'):
    post_train, post_test = load_cache('tfidf_char_union_extra_data_attack')
else:
    TFIDF_UNION1.update({'train': attack, 'test': test})
    post_trainw, post_testw = run_tfidf(**TFIDF_UNION1)
    TFIDF_UNION2.update({'train': attack, 'test': test})
    post_trainc, post_testc = run_tfidf(**TFIDF_UNION2)
    post_train = csr_matrix(hstack([post_trainw, post_trainc]))
    del post_trainw; del post_trainc; gc.collect()
    post_test = csr_matrix(hstack([post_testw, post_testc]))
    del post_testw; del post_testc; gc.collect()
    save_in_cache('tfidf_char_union_extra_data_attack', post_train, post_test)


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run TFIDF WORD-CHAR UNION 2/2')
if is_in_cache('tfidf_char_union_extra_data_toxic'):
    post_train, post_test = load_cache('tfidf_char_union_extra_data_toxic')
else:
    TFIDF_UNION1.update({'train': toxic, 'test': test})
    post_trainw, post_testw = run_tfidf(**TFIDF_UNION1)
    TFIDF_UNION2.update({'train': toxic, 'test': test})
    post_trainc, post_testc = run_tfidf(**TFIDF_UNION2)
    post_train = csr_matrix(hstack([post_trainw, post_trainc]))
    del post_trainw; del post_trainc; gc.collect()
    post_test = csr_matrix(hstack([post_testw, post_testc]))
    del post_testw; del post_testc; gc.collect()
    save_in_cache('tfidf_char_union_extra_data_toxic', post_train, post_test)


print('~~~~~~~~~~~~~~~~~~')
print_step('Run Attack LR')
train, test = run_cv_model(label='extra_data_attack_lr',
                           data_key='tfidf_char_union_extra_data_attack',
                           train_key='extra_data_attack',
                           model_fn=runSagLR,
                           train=attack,
                           test=test,
                           targets=['attack', 'quoting_attack', 'recipient_attack',
                                    'third_party_attack', 'other_attack'],
                           kf=kf)


print('~~~~~~~~~~~~~~~~~~~~~')
print_step('Run Attack Ridge')
train, test = run_cv_model(label='extra_data_attack_lr',
                           data_key='tfidf_char_union_extra_data_attack',
                           train_key='extra_data_attack',
                           model_fn=runRidge,
                           train=attack,
                           test=test,
                           targets=['attack_score'],
                           eval_fn=rmse,
                           kf=kf_for_regression)


print('~~~~~~~~~~~~~~~~~')
print_step('Run Toxic LR')
train, test = run_cv_model(label='extra_data_toxic_lr',
                           data_key='tfidf_char_union_extra_data_toxic',
                           train_key='extra_data_toxic',
                           model_fn=runSagLR,
                           train=toxic,
                           test=test,
                           targets=['toxicity_label'],
                           kf=kf)


print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Toxic Ridge')
train, test = run_cv_model(label='extra_data_toxic_lr',
                           data_key='tfidf_char_union_extra_data_toxic',
                           train_key='extra_data_toxic',
                           model_fn=runRidge,
                           train=toxic,
                           test=test,
                           targets=['toxicity'],
                           eval_fn=rmse,
                           kf=kf_for_regression)

print('~~~~~~~~~~~~')
print_step('Loading')
train_base, test_base = get_data()
print_step('Merging')
train_ = train_base.merge(test, on='id')
test_ = test_base.merge(test, on='id')
print_step('Dropping')
train_.drop(['id', 'comment_text_x', 'comment_text_y'], axis=1, inplace=True)
test_.drop(['id', 'comment_text_x', 'comment_text_y'], axis=1, inplace=True)

print_step('Scoring')
print('Toxic - Toxicity Score AUC: ' + str(roc_auc_score(train_['toxic'], train_['extra_data_toxic_lr_toxicity'])))
print('Toxic - Toxicity Label AUC: ' + str(roc_auc_score(train_['toxic'], train_['extra_data_toxic_lr_toxicity_label'])))
print('Severe Toxic - Toxicity Score AUC: ' + str(roc_auc_score(train_['severe_toxic'], train_['extra_data_toxic_lr_toxicity'])))
print('Severe Toxic - Toxicity Label AUC: ' + str(roc_auc_score(train_['severe_toxic'], train_['extra_data_toxic_lr_toxicity_label'])))
print('Threat - Attack Score AUC: ' + str(roc_auc_score(train_['threat'], train_['extra_data_attack_lr_attack_score'])))
print('Threat - Attack Label AUC: ' + str(roc_auc_score(train_['threat'], train_['extra_data_attack_lr_attack'])))
print('Threat - Quoting Attack AUC: ' + str(roc_auc_score(train_['threat'], train_['extra_data_attack_lr_quoting_attack'])))
print('Threat - Recipient Attack AUC: ' + str(roc_auc_score(train_['threat'], train_['extra_data_attack_lr_recipient_attack'])))
print('Threat - Third Party Attack AUC: ' + str(roc_auc_score(train_['threat'], train_['extra_data_attack_lr_third_party_attack'])))
print('Threat - Other Attack AUC: ' + str(roc_auc_score(train_['threat'], train_['extra_data_attack_lr_other_attack'])))
print('Insult - Attack Score AUC: ' + str(roc_auc_score(train_['insult'], train_['extra_data_attack_lr_attack_score'])))
print('Insult - Attack Label AUC: ' + str(roc_auc_score(train_['insult'], train_['extra_data_attack_lr_attack'])))
print('Insult - Quoting Attack AUC: ' + str(roc_auc_score(train_['insult'], train_['extra_data_attack_lr_quoting_attack'])))
print('Insult - Recipient Attack AUC: ' + str(roc_auc_score(train_['insult'], train_['extra_data_attack_lr_recipient_attack'])))
print('Insult - Third Party Attack AUC: ' + str(roc_auc_score(train_['insult'], train_['extra_data_attack_lr_third_party_attack'])))
print('Insult - Other Attack AUC: ' + str(roc_auc_score(train_['insult'], train_['extra_data_attack_lr_other_attack'])))

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('extra_data', train_, test_)
print_step('Done!')
