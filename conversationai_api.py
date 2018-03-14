import os
import requests
import json
import time

import pandas as pd

from sklearn.metrics import roc_auc_score

from preprocess import normalize_text
from cache import get_data, is_in_cache, load_cache, save_in_cache
from utils import print_step


CONVAI_KEY = os.environ['CONVAI_KEY']


train, test = get_data()
train['is_train'] = 1
test['is_train'] = 0
merge = pd.concat([train.reset_index(), test.reset_index()]).drop('index', axis=1)


def run_query(comment_text, idx):
    value = normalize_text(comment_text)
    value = value[:2999] if len(value) >= 3000 else value
    value = 'empty' if len(value) == 0 else value
    try:
        rr = requests.post('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze', params={'key': CONVAI_KEY}, data=json.dumps({'comment': {'text': value}, 'languages': ['en'], 'requestedAttributes': {'TOXICITY': {}, 'ATTACK_ON_AUTHOR': {}, 'ATTACK_ON_COMMENTER': {}, 'INCOHERENT': {}, 'INFLAMMATORY': {}, 'LIKELY_TO_REJECT': {}, 'OBSCENE': {}, 'SEVERE_TOXICITY': {}, 'SPAM': {}, 'UNSUBSTANTIAL': {}}}))
        return [(k, v['summaryScore']['value']) for k, v in rr.json()['attributeScores'].items()] + [('id', idx)]
    except Exception as error:
        print_step('FATAL ABORT:')
        import pdb
        pdb.set_trace()


def run_query_in_batches(df, label=''):
    responses = []
    total = len(df.comment_text.values)
    i = 0
    while i <= total:
        skip = False
        if i % 500 == 0 or i == total:
            batch_num = str(i / 500 + 1)
            if is_in_cache('convai-batches-' + label + batch_num):
                print_step('BATCH ' + label + batch_num + ' ALREADY DONE...')
                i += 500
                skip = True
            elif len(responses) > 100:
                batch_num = str(i / 500)
                if i == total:
                    batch_num = str(int(batch_num) + 1)
                    skip = True
                print_step('COLLECTING BATCH ' + label + batch_num + ' / ' + str(round(total / 500) + 1))
                batch_df = pd.DataFrame([dict(x) for x in responses])
                save_in_cache('convai-batches-' + label + batch_num, batch_df, None)
                batch_num = str(i / 500 + 1)
                print_step('SLEEPING 60s')
                time.sleep(60)
                responses = []
                print_step('STARTING BATCH ' + label + batch_num)
            else:
                print_step('STARTING BATCH ' + label + batch_num)
        if not skip:
            print_step(str(i + 1) + ' / ' + str(total))
            responses.append(run_query(df.comment_text.values[i], df.id.values[i]))
            i += 1

run_query_in_batches(merge)

dfs = []
for i in range(1, 627):
    df = pd.read_csv('cache/train_convai-batches-' + str(i) + '.csv')
    df['batch_num'] = i
    dfs.append(df)

df = pd.concat(dfs)
merge2 = pd.merge(df.reset_index(), merge.reset_index(), on='id').drop(['index_x', 'index_y'], axis=1)
print('DUPLICATION CHECK')
print(merge2.duplicated('id').value_counts())
print('TOXIC AUC:'  + str(roc_auc_score(merge2[merge2['is_train'] == 1]['toxic'], merge2[merge2['is_train'] == 1]['TOXICITY'])))
print('SEVTOX AUC: ' + str(roc_auc_score(merge2[merge2['is_train'] == 1]['severe_toxic'], merge2[merge2['is_train'] == 1]['SEVERE_TOXICITY'])))
print('OBSCENE AUC: ' + str(roc_auc_score(merge2[merge2['is_train'] == 1]['obscene'], merge2[merge2['is_train'] == 1]['TOXICITY'])))
print('THREAT AUC: ' + str(roc_auc_score(merge2[merge2['is_train'] == 1]['threat'], merge2[merge2['is_train'] == 1]['SEVERE_TOXICITY'])))
print('INSULT AUC: ' + str(roc_auc_score(merge2[merge2['is_train'] == 1]['insult'], merge2[merge2['is_train'] == 1]['TOXICITY'])))
print('IDENTITY HATE AUC: ' + str(roc_auc_score(merge2[merge2['is_train'] == 1]['identity_hate'], merge2[merge2['is_train'] == 1]['SEVERE_TOXICITY'])))

missing_ids = set(merge['id'].values) - set(merge2['id'].values)
print('# MISSING IDS: ' + str(len(missing_ids)))
missing_df = merge[merge['id'].apply(lambda x: x in missing_ids)]
import pdb
pdb.set_trace()
# extra_df = pd.DataFrame(dict(run_query(missing_df.comment_text.values[0], missing_df.id.values[0])), index=[1])
# df2 = pd.concat([df, extra_df])
# merge2 = pd.merge(df2.reset_index(), merge.reset_index(), on='id').drop(['index_x', 'index_y'], axis=1)

print('~~~~~~~~~~~')
print_step('Saving')
train2 = merge2[merge2['is_train'] == 1]
test2 = merge2[merge2['is_train'] == 0]
train2.drop(['comment_text', 'batch_num', 'is_train'], axis=1, inplace=True)
test2.drop(['comment_text', 'batch_num', 'is_train'], axis=1, inplace=True)
print('TOXIC AUC RECHECK: ' + str(roc_auc_score(train2['toxic'], train2['TOXICITY'])))
print('SEVTOX AUC RECHECK: ' + str(roc_auc_score(train2['severe_toxic'], train2['SEVERE_TOXICITY'])))
print('OBSCENE AUC RECHECK: ' + str(roc_auc_score(train2['obscene'], train2['TOXICITY'])))
print('THREAT AUC RECHECK: ' + str(roc_auc_score(train2['threat'], train2['SEVERE_TOXICITY'])))
print('INSULT AUC RECHECK: ' + str(roc_auc_score(train2['insult'], train2['TOXICITY'])))
print('IHATE AUC RECHECK: ' + str(roc_auc_score(train2['identity_hate'], train2['SEVERE_TOXICITY'])))
save_in_cache('convai_data', train2, test2)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test2['id']
submission['toxic'] = test2['TOXICITY']
submission['severe_toxic'] = test2['SEVERE_TOXICITY']
submission['obscene'] = test2['TOXICITY']
submission['threat'] = test2['SEVERE_TOXICITY']
submission['insult'] = test2['TOXICITY']
submission['identity_hate'] = test2['SEVERE_TOXICITY']
submission.to_csv('submit/submit_convai.csv', index=False)
print_step('Done!')
