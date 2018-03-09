import pandas as pd

from utils import print_step


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Stage 3 Data')
lgb_submit = pd.read_csv('ec2submit/submit_lvl2_lgb2.csv')
slgb_submit = pd.read_csv('ec2submit/submit_lvl2_sparse_lgb.csv')
gru_submit = pd.read_csv('submit/submit_lvl3_gru_4x_average.csv')
import pdb
pdb.set_trace()


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = lgb_submit['id']
submission['toxic'] = lgb_submit['toxic'] * 0.5 + slgb_submit['toxic'] * 0.1 + gru_submit['toxic'] * 0.4
submission['severe_toxic'] = lgb_submit['severe_toxic'] * 0.5 + slgb_submit['severe_toxic'] * 0.1 + gru_submit['severe_toxic'] * 0.4
submission['obscene'] = lgb_submit['obscene'] * 0.5 + slgb_submit['obscene'] * 0.1 + gru_submit['obscene'] * 0.4
submission['threat'] = lgb_submit['threat'] * 0.5 + slgb_submit['threat'] * 0.1 + gru_submit['threat'] * 0.4
submission['insult'] = lgb_submit['insult'] * 0.5 + slgb_submit['insult'] * 0.1 + gru_submit['insult'] * 0.4
submission['identity_hate'] = lgb_submit['identity_hate'] * 0.5 + slgb_submit['identity_hate'] * 0.1 + gru_submit['identity_hate'] * 0.4
submission.to_csv('submit/submit_lvl3_lgb2_gru4x_average.csv', index=False)
print_step('Done')
