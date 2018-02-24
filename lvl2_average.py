import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from utils import print_step
from preprocess import get_stage2_data


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Stage 2 Data')
train, test = get_stage2_data()


print('~~~~~~~~~~~~')
print_step('Scoring')
toxic = train['tfidf_union_nblr_toxic'] * 0.4 + train['tfidf_char_nblr_toxic'] * 0.4 + train['tfidf_union_lr_toxic'] * 0.1 + train['tfidf_char_lr_toxic'] * 0.1
print('Toxic AUC: ', roc_auc_score(train['toxic'], toxic))
severe_toxic = train['tfidf_union_nblr_severe_toxic'] * 0.4 + train['tfidf_char_nblr_severe_toxic'] * 0.4 + train['tfidf_union_lr_severe_toxic'] * 0.1 + train['tfidf_char_lr_severe_toxic'] * 0.1
print('Severe Toxic AUC: ', roc_auc_score(train['severe_toxic'], severe_toxic))
obscene = train['tfidf_union_nblr_obscene'] * 0.4 + train['tfidf_char_nblr_obscene'] * 0.4 + train['tfidf_union_lr_obscene'] * 0.1 + train['tfidf_char_lr_obscene'] * 0.1
print('Obscene AUC: ', roc_auc_score(train['obscene'], obscene))
threat = train['tfidf_union_nblr_threat'] * 0.4 + train['tfidf_char_nblr_threat'] * 0.4 + train['tfidf_union_lr_threat'] * 0.1 + train['tfidf_char_lr_threat'] * 0.1
print('Threat AUC: ', roc_auc_score(train['threat'], threat))
insult = train['tfidf_union_nblr_insult'] * 0.4 + train['tfidf_char_nblr_insult'] * 0.4 + train['tfidf_union_lr_insult'] * 0.1 + train['tfidf_char_lr_insult'] * 0.1
print('Insult AUC: ', roc_auc_score(train['insult'], insult))
identity_hate = train['tfidf_union_nblr_identity_hate'] * 0.4 + train['tfidf_char_nblr_identity_hate'] * 0.4 + train['tfidf_union_lr_identity_hate'] * 0.1 + train['tfidf_char_lr_identity_hate'] * 0.1
print('Identity Hate AUC: ', roc_auc_score(train['identity_hate'], identity_hate))
print('Overall AUC: ', np.mean([roc_auc_score(train['toxic'], toxic),
                                roc_auc_score(train['severe_toxic'], severe_toxic),
                                roc_auc_score(train['obscene'], obscene),
                                roc_auc_score(train['threat'], threat),
                                roc_auc_score(train['insult'], insult),
                                roc_auc_score(train['identity_hate'], identity_hate)]))


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['toxic'] = test['tfidf_union_nblr_toxic'] * 0.4 + test['tfidf_char_nblr_toxic'] * 0.4 + test['tfidf_union_lr_toxic'] * 0.1 + test['tfidf_char_lr_toxic'] * 0.1
submission['severe_toxic'] = test['tfidf_union_nblr_severe_toxic'] * 0.4 + test['tfidf_char_nblr_severe_toxic'] * 0.4 + test['tfidf_union_lr_severe_toxic'] * 0.1 + test['tfidf_char_lr_severe_toxic'] * 0.1
submission['obscene'] = test['tfidf_union_nblr_obscene'] * 0.4 + test['tfidf_char_nblr_obscene'] * 0.4 + test['tfidf_union_lr_obscene'] * 0.1 + test['tfidf_char_lr_obscene'] * 0.1
submission['threat'] = test['tfidf_union_nblr_threat'] * 0.4 + test['tfidf_char_nblr_threat'] * 0.4 + test['tfidf_union_lr_threat'] * 0.1 + test['tfidf_char_lr_threat'] * 0.1
submission['insult'] = test['tfidf_union_nblr_insult'] * 0.4 + test['tfidf_char_nblr_insult'] * 0.4 + test['tfidf_union_lr_insult'] * 0.1 + test['tfidf_char_lr_insult'] * 0.1
submission['identity_hate'] = test['tfidf_union_nblr_identity_hate'] * 0.4 + test['tfidf_char_nblr_identity_hate'] * 0.4 + test['tfidf_union_lr_identity_hate'] * 0.1 + test['tfidf_char_lr_identity_hate'] * 0.1
submission.to_csv('submit/submit_lvl2_simple_average.csv', index=False)
print_step('Done')
import pdb
pdb.set_trace()

# ('Toxic AUC: ', 0.9821125401005304)
# ('Severe Toxic AUC: ', 0.98875585323622828)
# ('Obscene AUC: ', 0.99267802516742354)
# ('Threat AUC: ', 0.99072057324329676)
# ('Insult AUC: ', 0.98484978627026554)
# ('Identity Hate AUC: ', 0.98481727585365408)
# ('Overall AUC: ', 0.98732234231189986)
