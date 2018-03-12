import pandas as pd

from sklearn.metrics import roc_auc_score

from utils import print_step


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Stage 3 Data')
lr_lgb_train = pd.read_csv('cache/train_lvl2_lr_lgb.csv')
sparse_lgb_train = pd.read_csv('cache/train_lvl2_sparse_lgb.csv')
lr_lgb_submit = pd.read_csv('submit/submit_lvl2_lgb2.csv')
sparse_lgb_submit = pd.read_csv('submit/submit_lvl2_sparse_lgb.csv')

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = lr_lgb_submit['id']

print('toxic AUC: ' + str(roc_auc_score(lr_lgb_train['toxic'], lr_lgb_train['lvl2_lgb_toxic'] * 0.8 + sparse_lgb_train['lvl2_sparse_lgb_toxic'] * 0.2)))
print('severe_toxic AUC: ' + str(roc_auc_score(lr_lgb_train['severe_toxic'], lr_lgb_train['lvl2_lgb_severe_toxic'] * 0.75 + sparse_lgb_train['lvl2_sparse_lgb_severe_toxic'] * 0.25)))
print('obscene AUC: ' + str(roc_auc_score(lr_lgb_train['obscene'], lr_lgb_train['lvl2_lgb_obscene'] * 0.6 + sparse_lgb_train['lvl2_sparse_lgb_obscene'] * 0.4)))
print('threat AUC: ' + str(roc_auc_score(lr_lgb_train['threat'], lr_lgb_train['lvl2_lgb_threat'] * 0.4 + sparse_lgb_train['lvl2_sparse_lgb_threat'] * 0.6)))
print('insult AUC: ' + str(roc_auc_score(lr_lgb_train['insult'], lr_lgb_train['lvl2_lgb_insult'] * 0.75 + sparse_lgb_train['lvl2_sparse_lgb_insult'] * 0.25)))
print('identity_hate AUC: ' + str(roc_auc_score(lr_lgb_train['identity_hate'], lr_lgb_train['lvl2_lgb_identity_hate'] * 0.7 + sparse_lgb_train['lvl2_sparse_lgb_identity_hate'] * 0.3)))

import pdb
pdb.set_trace()
submission['toxic'] = lr_lgb_submit['toxic'] * 0.8 + sparse_lgb_submit['toxic'] * 0.2
submission['severe_toxic'] = lr_lgb_submit['severe_toxic'] * 0.75 + sparse_lgb_submit['severe_toxic'] * 0.25
submission['obscene'] = lr_lgb_submit['obscene'] * 0.6 + sparse_lgb_submit['obscene'] * 0.4
submission['threat'] = lr_lgb_submit['threat'] * 0.4 + sparse_lgb_submit['threat'] * 0.6
submission['insult'] = lr_lgb_submit['insult'] * 0.75 + sparse_lgb_submit['insult'] * 0.25
submission['identity_hate'] = lr_lgb_submit['identity_hate'] * 0.7 + sparse_lgb_submit['identity_hate'] * 0.3
submission.to_csv('submit/submit_lvl3_lgb_average.csv', index=False)
print_step('Done')
# toxic AUC: 0.9856267755121848
# severe_toxic AUC: 0.9913076713529598
# obscene AUC: 0.9945214443455751
# threat AUC: 0.9918026421061
# insult AUC: 0.9878876692119437
# identity_hate AUC: 0.9885451106979231
