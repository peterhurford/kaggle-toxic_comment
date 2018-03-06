import pandas as pd

from utils import print_step


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Stage 3 Data')
gru_6 = pd.read_csv('submit/submit_kernel_gru_v6.csv')
gru_6_3 = pd.read_csv('submit/submit_kernel_gru_v6_3epic.csv')
gru_9 = pd.read_csv('submit/submit_kernel_gru_v9.csv')
gru_9_3 = pd.read_csv('submit/submit_kernel_gru_v9_3epic.csv')
import pdb
pdb.set_trace()


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = gru_6['id']
submission['toxic'] = gru_6['toxic'] * 0.25 + gru_6_3['toxic'] * 0.25 + gru_9['toxic'] * 0.25 + gru_9_3['toxic'] * 0.25
submission['severe_toxic'] = gru_6['severe_toxic'] * 0.25 + gru_6_3['severe_toxic'] * 0.25 + gru_9['severe_toxic'] * 0.25 + gru_9_3['severe_toxic'] * 0.25
submission['obscene'] = gru_6['obscene'] * 0.25 + gru_6_3['obscene'] * 0.25 + gru_9['obscene'] * 0.25 + gru_9_3['obscene'] * 0.25
submission['threat'] = gru_6['threat'] * 0.25 + gru_6_3['threat'] * 0.25 + gru_9['threat'] * 0.25 + gru_9_3['threat'] * 0.25
submission['insult'] = gru_6['insult'] * 0.25 + gru_6_3['insult'] * 0.25 + gru_9['insult'] * 0.25 + gru_9_3['insult'] * 0.25
submission['identity_hate'] = gru_6['identity_hate'] * 0.25 + gru_6_3['identity_hate'] * 0.25 + gru_9['identity_hate'] * 0.25 + gru_9_3['identity_hate'] * 0.25
submission.to_csv('submit/submit_lvl3_gru_4x_average.csv', index=False)
print_step('Done')
