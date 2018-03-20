import gc
import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from cv import run_cv_model
from utils import print_step
from cache import is_in_cache, load_cache, save_in_cache, get_data
from feature_engineering import add_features


def runLGB(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    rounds_lookup = {'toxic': 1000,
                     'severe_toxic': 2500,
                     'obscene': 1750,
                     'threat': 450,
                     'insult': 850,
                     'identity_hate': 750}
    leaves_lookup = {'toxic': 15,
                     'severe_toxic': 4,
                     'obscene': 4,
                     'threat': 15,
                     'insult': 15,
                     'identity_hate': 15}
    learning_rate_lookup = {'toxic': 0.02,
                            'severe_toxic': 0.005,
                            'obscene': 0.005,
                            'threat': 0.02,
                            'insult': 0.02,
                            'identity_hate': 0.02}
    params = {
        'learning_rate': learning_rate_lookup[label],
        'application': 'binary',
        'num_leaves': leaves_lookup[label],
        'verbosity': -1,
        'metric': 'auc',
        'data_random_seed': 12,
        'bagging_fraction': 1.0,
        'feature_fraction': 0.2,
        'nthread': min(mp.cpu_count() - 1, 6),
    }
    model = lgb.train(params,
                      train_set=d_train,
                      #num_boost_round=10000,
                      num_boost_round=rounds_lookup[label],
                      valid_sets=watchlist,
                      verbose_eval=50)
                      #early_stopping_rounds=800)
    #lgb_eval = model.best_score['valid_1']['auc']
    #best_round = model.best_iteration
    #print('[LGB %s] Found %.6f @ %d / 10000' % (label, lgb_eval, best_round))
    print(model.feature_importance())
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    pred_test_y = minmax_scale(pd.Series(pred_test_y).rank().values) # Rank transform
    pred_test_y2 = minmax_scale(pd.Series(pred_test_y2).rank().values)
    return pred_test_y, pred_test_y2

def runXGB(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    rounds_lookup = {'toxic': 700,
                     'severe_toxic': 1000,
                     'obscene': 700,
                     'threat': 400,
                     'insult': 1000,
                     'identity_hate': 750}
    max_depth_lookup = {'toxic': 8,
                        'severe_toxic': 3,
                        'obscene': 3,
                        'threat': 8,
                        'insult': 8,
                        'identity_hate': 8}
    learning_rate_lookup = {'toxic': 0.005,
                            'severe_toxic': 0.01,
                            'obscene': 0.01,
                            'threat': 0.005,
                            'insult': 0.005,
                            'identity_hate': 0.005}
    params = {
        #'n_estimators': 10000,
        'n_estimators': rounds_lookup[label],
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': learning_rate_lookup[label],
        'colsample_bytree': 0.1, # 0.4
        'max_depth': max_depth_lookup[label],
        'scale_pos_weight': 5, # 10
        'gamma': 0.01,
        'nthread': min(mp.cpu_count() - 1, 6),
        'seed': 14
    }
    model = XGBClassifier(**params)
    model.fit(train_X, train_y,
              eval_set=[(train_X, train_y), (test_X, test_y)],
              verbose=10)
              #early_stopping_rounds=200)
    #xgb_eval = model.evals_result_['validation_1']['auc']
    #best_round = np.argsort(xgb_eval)[-1]
    #best_eval = xgb_eval[best_round]
    #print('[XGB %s] Found %.6f @ %d / 10000' % (label, best_eval, best_round))
    #pred_test_y = model.predict_proba(test_X, ntree_limit=best_round)[:, 1]
    #pred_test_y2 = model.predict_proba(test_X2, ntree_limit=best_round)[:, 1]
    pred_test_y = model.predict_proba(test_X)[:, 1]
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    pred_test_y = minmax_scale(pd.Series(pred_test_y).rank().values) # Rank transform
    pred_test_y2 = minmax_scale(pd.Series(pred_test_y2).rank().values)
    return pred_test_y, pred_test_y2

def runRF(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    max_features_lookup = {'toxic': 0.1,
                           'severe_toxic': 0.3,
                           'obscene': 0.3,
                           'threat': 0.1,
                           'insult': 0.2,
                           'identity_hate': 0.1}
    max_leaf_nodes_lookup = {'toxic': 200,
                             'severe_toxic': 200,
                             'obscene': 100,
                             'threat': 100,
                             'insult': 500,
                             'identity_hate': 200}
    params = {
        'n_estimators': 500,
        'criterion': 'entropy',
        'max_features': max_features_lookup[label],
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_leaf_nodes': max_leaf_nodes_lookup[label],
        'class_weight': 'balanced_subsample',
        'n_jobs': min(mp.cpu_count() - 1, 6),
        'random_state': 16,
        'verbose': 2
    }
    model = RandomForestClassifier(**params)
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)[:, 1]
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    pred_test_y = minmax_scale(pd.Series(pred_test_y).rank().values) # Rank transform
    pred_test_y2 = minmax_scale(pd.Series(pred_test_y2).rank().values)
    return pred_test_y, pred_test_y2


if not is_in_cache('lvl2_all'):
    print_step('Importing 1/21: LRs')
    lr_train, lr_test = load_cache('lvl1_lr')
    print_step('Importing 2/21: FE')
    train_fe, test_fe = load_cache('fe_lgb_data')
    print_step('Importing 3/21: Sparse LGBs')
    lgb_train, lgb_test = load_cache('lvl1_sparse_lgb')
    print_step('Importing 4/21: FE LGB')
    fe_lgb_train, fe_lgb_test = load_cache('lvl1_fe_lgb')
    print_step('Importing 5/21: Sparse FE LGB')
    sfe_lgb_train, sfe_lgb_test = load_cache('lvl1_sparse_fe_lgb')
    print_step('Importing 6/21: FM')
    fm_train, fm_test = load_cache('lvl1_fm')
    print_step('IMporting 7/21: Ridge')
    ridge_train, ridge_test = load_cache('lvl1_ridge')
    print_step('Importing 8/21: GRU')
    gru_train, gru_test = load_cache('lvl1_gru')
    print_step('Importing 9/21: GRU2')
    gru2_train, gru2_test = load_cache('lvl1_gru2')
    print_step('Importing 10/21: GRU80')
    gru80_train, gru80_test = load_cache('lvl1_gru80')
    print_step('Importing 11/21: GRU-Conv')
    gru_conv_train, gru_conv_test = load_cache('lvl1_gru-conv')
    print_step('Importing 12/21: GRU128')
    gru128_train, gru128_test = load_cache('lvl1_gru128')
    print_step('Importing 13/21: GRU128-2')
    gru128_2_train, gru128_2_test = load_cache('lvl1_gru128-2')
    print_step('Importing 14/21: 2DConv')
    x2dconv_train, x2dconv_test = load_cache('lvl1_2dconv')
    print_step('Importing 15/21: LSTMConv')
    lstm_conv_train, lstm_conv_test = load_cache('lvl1_lstm-conv')
    print_step('Importing 16/21: DPCNN')
    dpcnn_train, dpcnn_test = load_cache('lvl1_dpcnn')
    print_step('Importing 17/21: RNNCNN')
    rnncnn_train, rnncnn_test = load_cache('lvl1_rnncnn')
    print_step('Importing 18/21: RNNCNN2')
    rnncnn2_train, rnncnn2_test = load_cache('lvl1_rnncnn2')
    print_step('Importing 19/21: CapsuleNet')
    capsule_net_train, capsule_net_test = load_cache('lvl1_capsule_net')
    print_step('Importing 20/21: Attention LSTM')
    attention_train, attention_test = load_cache('lvl1_attention-lstm')
    print_step('Importing 21/21: Neptune Models')
    neptune_train, neptune_test = load_cache('neptune_models')

    print_step('Merging 1/2')
    lr_keep = [c for c in lr_train.columns if 'lr' in c]
    lgb_keep = [c for c in lgb_train.columns if 'lgb' in c]
    fe_lgb_keep = [c for c in fe_lgb_train.columns if 'lgb' in c]
    sfe_lgb_keep = [c for c in sfe_lgb_train.columns if 'lgb' in c]
    fm_keep = [c for c in fm_train.columns if 'fm' in c]
    ridge_keep = [c for c in ridge_train.columns if 'ridge' in c]
    gru_keep = [c for c in gru_train.columns if 'gru' in c]
    gru2_keep = [c for c in gru2_train.columns if 'gru' in c]
    gru80_keep = [c for c in gru80_train.columns if 'gru' in c]
    gru_conv_keep = [c for c in gru_conv_train.columns if 'gru' in c]
    gru128_keep = [c for c in gru128_train.columns if 'gru' in c]
    gru128_2_keep = [c for c in gru128_2_train.columns if 'gru' in c]
    x2dconv_keep = [c for c in x2dconv_train.columns if 'conv' in c]
    lstm_conv_keep = [c for c in lstm_conv_train.columns if 'lstm' in c]
    dpcnn_keep = [c for c in dpcnn_train.columns if 'dpcnn' in c]
    rnncnn_keep = [c for c in rnncnn_train.columns if 'rnncnn' in c]
    rnncnn2_keep = [c for c in rnncnn2_train.columns if 'rnncnn' in c]
    capsule_net_keep = [c for c in capsule_net_train.columns if 'capsule' in c]
    attention_keep = [c for c in attention_train.columns if 'attention' in c]
    neptune_keep = [c for c in neptune_train.columns if 'neptune' in c]
    train_ = pd.concat([lr_train[lr_keep],
                        train_fe,
                        lgb_train[lgb_keep],
                        fe_lgb_train[fe_lgb_keep],
                        sfe_lgb_train[sfe_lgb_keep],
                        fm_train[fm_keep],
                        ridge_train[ridge_keep],
                        gru_train[gru_keep],
                        gru2_train[gru2_keep],
                        gru80_train[gru80_keep],
                        gru_conv_train[gru_conv_keep],
                        gru128_train[gru128_keep],
                        gru128_2_train[gru128_2_keep],
                        x2dconv_train[x2dconv_keep],
                        lstm_conv_train[lstm_conv_keep],
                        dpcnn_train[dpcnn_keep],
                        rnncnn_train[rnncnn_keep],
                        rnncnn2_train[rnncnn2_keep],
                        capsule_net_train[capsule_net_keep],
                        attention_train[attention_keep],
                        neptune_train[neptune_keep]], axis=1)
    print_step('Merging 2/2')
    test_ = pd.concat([lr_test[lr_keep],
                       test_fe,
                       lgb_test[lgb_keep],
                       fe_lgb_test[fe_lgb_keep],
                       sfe_lgb_test[sfe_lgb_keep],
                       fm_test[fm_keep],
                       ridge_test[ridge_keep],
                       gru_test[gru_keep],
                       gru2_test[gru2_keep],
                       gru80_test[gru80_keep],
                       gru_conv_test[gru_conv_keep],
                       gru128_test[gru128_keep],
                       gru128_2_test[gru128_2_keep],
                       x2dconv_test[x2dconv_keep],
                       lstm_conv_test[lstm_conv_keep],
                       dpcnn_test[dpcnn_keep],
                       rnncnn_test[rnncnn_keep],
                       rnncnn2_test[rnncnn2_keep],
                       capsule_net_test[capsule_net_keep],
                       attention_test[attention_keep],
                       neptune_test[neptune_keep]], axis=1)
    del lr_train
    del lr_test
    del train_fe
    del test_fe
    del lgb_train
    del lgb_test
    del fm_train
    del fm_test
    del ridge_train
    del ridge_test
    del gru_train
    del gru_test
    del gru80_train
    del gru80_test
    del gru128_train
    del gru128_test
    del gru128_2_train
    del gru128_2_test
    del gru_conv_train
    del gru_conv_test
    del x2dconv_train
    del x2dconv_test
    del lstm_conv_train
    del lstm_conv_test
    del dpcnn_train
    del dpcnn_test
    del rnncnn_train
    del rnncnn_test
    del rnncnn2_train
    del rnncnn2_test
    del capsule_net_train
    del capsule_net_test
    del attention_train
    del attention_test
    del neptune_train
    del neptune_test
    gc.collect()
    print('Train shape: {}'.format(train_.shape))
    print('Test shape: {}'.format(test_.shape))
    print_step('Saving')
    save_in_cache('lvl2_all', train_, test_)
else:
    train_, test_ = load_cache('lvl2_all')


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Level 2 LGB')
print(train_.columns.values)
train, test = get_data()
train_, test_ = run_cv_model(label='lvl2_all_lgb',
                             data_key='lvl2_all',
                             model_fn=runLGB,
                             train=train,
                             test=test,
                             kf=kf)
import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test_['id']
submission['toxic'] = test_['lvl2_all_lgb_toxic']
submission['severe_toxic'] = test_['lvl2_all_lgb_severe_toxic']
submission['obscene'] = test_['lvl2_all_lgb_obscene']
submission['threat'] = test_['lvl2_all_lgb_threat']
submission['insult'] = test_['lvl2_all_lgb_insult']
submission['identity_hate'] = test_['lvl2_all_lgb_identity_hate']
submission.to_csv('submit/submit_lvl2_all_lgb.csv', index=False)
print_step('Done')
# toxic CV scores : [0.9887967332779701, 0.9895775830168592, 0.9884461592833008, 0.9875096985077569, 0.9891583943034448]
# toxic mean CV : 0.9886977136778663 overall: 0.9886305781002419
# severe_toxic CV scores : [0.9926544211580293, 0.9913160339941093, 0.9923913598883994, 0.9935268119583621, 0.9916422631452835]
# severe_toxic mean CV : 0.9923061780288368 overall: 0.9922663186170257
# obscene CV scores : [0.9960631359478467, 0.9958689904610882, 0.9955226321963657, 0.9952276258375334, 0.9953261649899813]
# obscene mean CV : 0.9956017098865632 overall: 0.9955885858106922
# threat CV scores : [0.9938640565909258, 0.9952494971557874, 0.9953346140146873, 0.996113421400002, 0.9890386441305978]
# threat mean CV : 0.9939200466584 overall: 0.9927828850507612
# insult CV scores : [0.989133305492269, 0.9900039870947296, 0.9900339287741642, 0.9914731575524615, 0.9905497113473908]
# insult mean CV : 0.9902388180522029 overall: 0.9902120727501941
# identity_hate CV scores : [0.9898368281166015, 0.992818099662353, 0.9894087810681963, 0.992869399754052, 0.993508850897071]
# identity_hate mean CV : 0.9916883918996546 overall: 0.9915218089485965
# lvl2_all_lgb CV mean : 0.9920754763672539, overall: 0.9918337082129186


print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Level 2 XGB')
train_, test_ = run_cv_model(label='lvl2_all_xgb',
                             data_key='lvl2_all',
                             model_fn=runXGB,
                             train=train,
                             test=test,
                             kf=kf)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test_['id']
submission['toxic'] = test_['lvl2_all_xgb_toxic']
submission['severe_toxic'] = test_['lvl2_all_xgb_severe_toxic']
submission['obscene'] = test_['lvl2_all_xgb_obscene']
submission['threat'] = test_['lvl2_all_xgb_threat']
submission['insult'] = test_['lvl2_all_xgb_insult']
submission['identity_hate'] = test_['lvl2_all_xgb_identity_hate']
submission.to_csv('submit/submit_lvl2_all_xgb.csv', index=False)
print_step('Done')
# toxic CV scores : [0.9885785403468411, 0.9891123200112236, 0.9881130013449466, 0.9871996068312614, 0.9889966283459425]
# toxic mean CV : 0.988400019376043 overall: 0.9883277183397865
# severe_toxic CV scores : [0.992769609739894, 0.9913405408676922, 0.9924638883280309, 0.9935939826199633, 0.9916459342154154]
# severe_toxic mean CV : 0.9923627911541992 overall: 0.9923261408065953
# obscene CV scores : [0.9960722980016739, 0.9959081249602342, 0.9954985124874312, 0.9951966539385606, 0.9953861277672205]
# obscene mean CV : 0.9956123434310241 overall: 0.9955963820731004
# threat CV scores : [0.9941770247336497, 0.9958692133630849, 0.9955965120420711, 0.9967019661164982, 0.9862940870940314]
# threat mean CV : 0.993727760669867 overall: 0.991861816462869
# insult CV scores : [0.988941072061676, 0.9898272615887176, 0.9900814238261089, 0.9914597848544254, 0.990655179161343]
# insult mean CV : 0.9901929442984543 overall: 0.9901370166973809
# identity_hate CV scores : [0.9900172727348406, 0.9926022117764536, 0.9898988881942627, 0.993156162766641, 0.9937237262811607]
# identity_hate mean CV : 0.9918796523506718 overall: 0.9913569994460074
# lvl2_all_xgb CV mean : 0.9920292518800432, overall: 0.9916010123042899


print('~~~~~~~~~~~~~~~~~~~')
print_step('Run Level 2 RF')
train_, test_ = run_cv_model(label='lvl2_all_rf',
                             data_key='lvl2_all',
                             model_fn=runRF,
                             train=train,
                             test=test,
                             kf=kf)
# toxic CV scores : [0.9885894726510229, 0.9890478477385832, 0.9880790477168565, 0.9871221603842731, 0.9887895647471248]
# toxic mean CV : 0.9883256186475722 overall: 0.9883243919137541
# severe_toxic CV scores : [0.9919870020450189, 0.990823713723998, 0.9921366669957401, 0.993257831657622, 0.9896180648400283]
# severe_toxic mean CV : 0.9915646558524814 overall: 0.9915626047240541
# obscene CV scores : [0.9960333984269849, 0.9958745307628682, 0.9952203625160928, 0.9951928754451966, 0.9953790070649905]
# obscene mean CV : 0.9955400348432265 overall: 0.9955346694616329
# threat CV scores : [0.990977121897818, 0.9901197332306695, 0.9949296541898447, 0.9969328847292661, 0.9841817441964331]
# threat mean CV : 0.9914282276488062 overall: 0.991402912751198
# insult CV scores : [0.9882158073164107, 0.9898398728881406, 0.989956005079114, 0.9911303016652566, 0.9907421673484038]
# insult mean CV : 0.989976830859465 overall: 0.9899738875335782
# identity_hate CV scores : [0.988078843048506, 0.9924706990413744, 0.986153362749136, 0.9913809658434762, 0.9936844637109788]
# identity_hate mean CV : 0.9903536668786943 overall: 0.9903329908398866
# lvl2_all_rf CV mean : 0.991198172455041, overall: 0.9911885762040175

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['id'] = test_['id']
submission['toxic'] = test_['lvl2_all_rf_toxic']
submission['severe_toxic'] = test_['lvl2_all_rf_severe_toxic']
submission['obscene'] = test_['lvl2_all_rf_obscene']
submission['threat'] = test_['lvl2_all_rf_threat']
submission['insult'] = test_['lvl2_all_rf_insult']
submission['identity_hate'] = test_['lvl2_all_rf_identity_hate']
submission.to_csv('submit/submit_lvl2_all_rf.csv', index=False)
print_step('Done')


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Merging Level 2 CNN')
train_cnn, test_cnn = load_cache('lvl2_final-cnn')
labels = ['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate']
aucs = []
for label in labels:
    train_['final-cnn_' + label] = train_cnn['final-cnn_' + label]
    auc = roc_auc_score(train_[label], train_['final-cnn_' + label])
    aucs.append(auc)
    print(label + ' AUC: ' + str(auc))
    test_['final-cnn_' + label] = test_cnn['final-cnn_' + label]
print('Total Mean AUC: ' + str(np.mean(aucs)))


print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 2')
save_in_cache('lvl3_all_mix', train, test)
print_step('Done!')

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Assessing correlations')
for label in labels:
    print(label)
    print(pd.DataFrame(np.corrcoef([train_['lvl2_all_lgb_' + label], train_['lvl2_all_xgb_' + label], train_['final-cnn_' + label], train_['lvl2_all_rf_' + label]]),
                       columns = ['lgb', 'xgb', 'cnn', 'rf'],
                       index = ['lgb', 'xgb', 'cnn', 'rf']))

print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Averaging for lvl3')
submission = pd.DataFrame()
submission['id'] = test_['id']

# Found weights through lvl3_hillclimb_average.py and then normalized to sum to 1
w1 = 0.713; w2 = 0.011; w3 = 0.142; w4 = 0.134
toxic_train = minmax_scale(train_['lvl2_all_lgb_toxic'].rank() * w1 + train_['lvl2_all_xgb_toxic'].rank() * w2 + train_['final-cnn_toxic'].rank() * w3 + train_['lvl2_all_rf_toxic'].rank() * w4)
submission['toxic'] = minmax_scale(test_['lvl2_all_lgb_toxic'].rank() * w1 + test_['lvl2_all_xgb_toxic'].rank() * w2 + test_['final-cnn_toxic'].rank() * w3 + test_['lvl2_all_rf_toxic'].rank() * w4)
print('toxic AUC: ' + str(roc_auc_score(train_['toxic'], toxic_train)))

w1 = 0.231; w2 = 0.443; w3 = 0.279; w4 = 0.047
severe_toxic_train = minmax_scale(train_['lvl2_all_lgb_severe_toxic'].rank() * w1 + train_['lvl2_all_xgb_severe_toxic'].rank() * w2 + train_['final-cnn_severe_toxic'].rank() * w3 + train_['lvl2_all_rf_severe_toxic'].rank() * w4)
submission['severe_toxic'] = minmax_scale(test_['lvl2_all_lgb_severe_toxic'].rank() * w1 + test_['lvl2_all_xgb_severe_toxic'].rank() * w2 + test_['final-cnn_severe_toxic'].rank() * w3 + test_['lvl2_all_rf_severe_toxic'].rank() * w4)
print('severe_toxic AUC: ' + str(roc_auc_score(train_['severe_toxic'], severe_toxic_train)))

w1 = 0.146; w2 = 0.200; w3 = 0.309; w4 = 0.345
obscene_train = minmax_scale(train_['lvl2_all_lgb_obscene'].rank() * w1 + train_['lvl2_all_xgb_obscene'].rank() * w2 + train_['final-cnn_obscene'].rank() * w3 + train_['lvl2_all_rf_obscene'].rank() * w4)
submission['obscene'] = minmax_scale(test_['lvl2_all_lgb_obscene'].rank() * w1 + test_['lvl2_all_xgb_obscene'].rank() * w2 + test_['final-cnn_obscene'].rank() * w3 + test_['lvl2_all_rf_obscene'].rank() * w4)
print('obscene AUC: ' + str(roc_auc_score(train_['obscene'], obscene_train)))

w1 = 0.371; w2 = 0.005; w3 = 0.250; w4 = 0.374
insult_train = minmax_scale(train_['lvl2_all_lgb_insult'].rank() * w1 + train_['lvl2_all_xgb_insult'].rank() * w2 + train_['final-cnn_insult'].rank() * w3 + train['lvl2_all_rf_insult'].rank() * w4)
submission['insult'] = minmax_scale(test_['lvl2_all_lgb_insult'].rank() * w1 + test_['lvl2_all_xgb_insult'].rank() * w2 + test_['final-cnn_insult'].rank() * w3 + test_['lvl2_all_rf_insult'].rank() * w4)
print('insult AUC: ' + str(roc_auc_score(train_['insult'], insult_train)))

w1 = 0.204; w2 = 0.524; w3 = 0.124; w4 = 0.148
threat_train = minmax_scale(train_['lvl2_all_lgb_threat'].rank() * w1 + train_['lvl2_all_xgb_threat'].rank() * w2 + train_['final-cnn_threat'].rank() * w3 + train_['lvl2_all_rf_threat'].rank() * w4)
submission['threat'] = minmax_scale(test_['lvl2_all_lgb_threat'].rank() * w1 + test_['lvl2_all_xgb_threat'].rank() * w2 + test_['final-cnn_threat'].rank() * w3 + test_['lvl2_all_rf_threat'].rank() * w4)
print('threat AUC: ' + str(roc_auc_score(train_['threat'], threat_train)))

w1 = 0.318; w2 = 0.412; w3 = 0.141; w4 = 0.129
identity_hate_train = minmax_scale(train_['lvl2_all_lgb_identity_hate'].rank() * w1 + train_['lvl2_all_xgb_identity_hate'].rank() * w2 + train_['final-cnn_identity_hate'].rank() * w3 + train_['lvl2_all_rf_identity_hate'].rank() * w4)
submission['identity_hate'] = minmax_scale(test_['lvl2_all_lgb_identity_hate'].rank() * w1 + test_['lvl2_all_xgb_identity_hate'].rank() * w2 + test_['final-cnn_identity_hate'].rank() * w3 + test_['lvl2_all_rf_identity_hate'].rank() * w4)
print('identity_hate AUC: ' + str(roc_auc_score(train_['identity_hate'], identity_hate_train)))
import pdb
pdb.set_trace()

print_step('Writing')
submission.to_csv('submit/submit_lvl3_all_mix.csv', index=False)
print_step('Done')
