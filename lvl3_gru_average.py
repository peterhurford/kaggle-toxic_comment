import pandas as pd

from utils import print_step


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Stage 3 Data')
gru_200 = pd.read_csv('submit/submit_kernel_gru_200_dim.csv')
gru_300_2 = pd.read_csv('submit/submit_kernel_gru_300_dim_2epic.csv')
gru_300_3 = pd.read_csv('submit/submit_kernel_gru_300_dim_3epic.csv')
gru_300_2ft = pd.read_csv('submit/submit_kernel_gru_300_dim_2epic_fasttext.csv')


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = gru_300_2ft['id']
submission['toxic'] = gru_300_2ft['toxic'] * 0.34 + gru_300_2['toxic'] * 0.33 + gru_300_3['toxic'] * 0.33
submission['severe_toxic'] = gru_300_2ft['severe_toxic'] * 0.34 + gru_300_2['severe_toxic'] * 0.33 + gru_300_3['severe_toxic'] * 0.33
submission['obscene'] = gru_300_2ft['obscene'] * 0.34 + gru_300_2['obscene'] * 0.33 + gru_300_3['obscene'] * 0.33
submission['threat'] = gru_300_2ft['threat'] * 0.34 + gru_300_2['threat'] * 0.33 + gru_300_3['threat'] * 0.33
submission['insult'] = gru_300_2ft['insult'] * 0.34 + gru_300_2['insult'] * 0.33 + gru_300_3['insult'] * 0.33
submission['identity_hate'] = gru_300_2ft['identity_hate'] * 0.34 + gru_300_2['identity_hate'] * 0.33 + gru_300_3['identity_hate'] * 0.33
submission.to_csv('submit/submit_lvl3_gru_3x_average.csv', index=False)
print_step('Done')
import pdb
pdb.set_trace()
