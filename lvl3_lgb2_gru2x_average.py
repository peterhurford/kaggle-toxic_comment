import pandas as pd

from utils import print_step


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Stage 3 Data')
lgb_submit = pd.read_csv('submit/submit_lvl2_lgb2.csv')
gru_submit = pd.read_csv('submit/submit_lvl3_gru_2x_average.csv')


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = lgb_submit['id']
submission['toxic'] = lgb_submit['toxic'] * 0.5 + gru_submit['toxic'] * 0.5
submission['severe_toxic'] = lgb_submit['severe_toxic'] * 0.5 + gru_submit['severe_toxic'] * 0.5
submission['obscene'] = lgb_submit['obscene'] * 0.5 + gru_submit['obscene'] * 0.5
submission['threat'] = lgb_submit['threat'] * 0.5 + gru_submit['threat'] * 0.5
submission['insult'] = lgb_submit['insult'] * 0.5 + gru_submit['insult'] * 0.5
submission['identity_hate'] = lgb_submit['identity_hate'] * 0.5 + gru_submit['identity_hate'] * 0.5
submission.to_csv('submit/submit_lvl3_lgb2_gru2x_average.csv', index=False)
print_step('Done')
import pdb
pdb.set_trace()
