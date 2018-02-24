import pandas as pd

from utils import print_step


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Stage 3 Data')
gru_200 = pd.read_csv('submit/submit_kernel_gru_200_dim.csv')
gru_300_2 = pd.read_csv('submit/submit_kernel_gru_300_dim_2epic.csv')
gru_300_3 = pd.read_csv('submit/submit_kernel_gru_300_dim_3epic.csv')


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = gru_200['id']
submission['toxic'] = gru_200['toxic'] * 0.2 + gru_300_2['toxic'] * 0.4 + gru_300_3['toxic'] * 0.4
submission['severe_toxic'] = gru_200['severe_toxic'] * 0.2 + gru_300_2['severe_toxic'] * 0.4 + gru_300_3['severe_toxic'] * 0.4
submission['obscene'] = gru_200['obscene'] * 0.2 + gru_300_2['obscene'] * 0.4 + gru_300_3['obscene'] * 0.4
submission['threat'] = gru_200['threat'] * 0.2 + gru_300_2['threat'] * 0.4 + gru_300_3['threat'] * 0.4
submission['insult'] = gru_200['insult'] * 0.2 + gru_300_2['insult'] * 0.4 + gru_300_3['insult'] * 0.4
submission['identity_hate'] = gru_200['identity_hate'] * 0.2 + gru_300_2['identity_hate'] * 0.4 + gru_300_3['identity_hate'] * 0.4
submission.to_csv('submit/submit_lvl3_gru_3x_average.csv', index=False)
print_step('Done')
import pdb
pdb.set_trace()
