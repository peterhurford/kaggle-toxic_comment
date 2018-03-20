import gc
import numpy as np

from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from cv import run_cv_model
from utils import print_step
from preprocess import run_tfidf, clean_text
from cache import get_data, is_in_cache, load_cache, save_in_cache


# TFIDF Hyperparams
TFIDF_PARAMS_WORD = {'ngram_min': 1,
                     'ngram_max': 2,
                     'min_df': 1,
                     'max_features': 200000,
                     'rm_stopwords': False,
                     'analyzer': 'word',
                     'tokenize': False,
                     'binary': True}

TFIDF_PARAMS_WORD_NOSTOP = {'ngram_min': 1,
                          'ngram_max': 2,
                          'min_df': 1,
                          'max_features': 200000,
                          'rm_stopwords': False,
                          'analyzer': 'word',
                          'tokenize': False,
                          'binary': True}

TFIDF_PARAMS_CHAR = {'ngram_min': 2,
                     'ngram_max': 6,
                     'min_df': 1,
                     'max_features': 200000,
                     'rm_stopwords': False,
                     'analyzer': 'char',
                     'tokenize': False,
                     'binary': True}

# Combine both word-level and character-level
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


# LR Model Definitions
def runSagLR(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    model = LogisticRegression(solver='sag')
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)[:, 1]
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2

def runL1LR(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    model = LogisticRegression(C=5, penalty='l1')
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)[:, 1]
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2

def runL2LR(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    model = LogisticRegression(C=5, penalty='l2')
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)[:, 1]
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2


def pr(x, y_i, y):
    p = x[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)

def runNBLR(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    train_y = train_y.values
    r = csr_matrix(np.log(pr(train_X, 1, train_y) / pr(train_X, 0, train_y)))
    model = LogisticRegression(C=4, dual=True)
    x_nb = train_X.multiply(r)
    model.fit(x_nb, train_y)
    pred_test_y = model.predict_proba(test_X.multiply(r))[:, 1]
    pred_test_y2 = model.predict_proba(test_X2.multiply(r))[:, 1]
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = get_data()
train['non_toxic'] = train[['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate']].sum(axis=1).apply(lambda x: 0 if x > 1 else 1)
save_in_cache('extra_label', train, test)


if not is_in_cache('cleaned'):
    print('~~~~~~~~~~~~~')
    print_step('Cleaning')
    train_cleaned, test_cleaned = clean_text(train, test)
    save_in_cache('cleaned', train_cleaned, test_cleaned)
else:
    train_cleaned, test_cleaned = load_cache('cleaned')


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Making KFold for CV')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


if not is_in_cache('tfidf_word'):
    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Run TFIDF WORD')
    TFIDF_PARAMS_WORD.update({'train': train, 'test': test})
    post_train, post_test = run_tfidf(**TFIDF_PARAMS_WORD)
    save_in_cache('tfidf_word', post_train, post_test)
    del post_train
    del post_test
    gc.collect()


if not is_in_cache('tfidf_word_nostop'):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Run TFIDF WORD NO STOP')
    TFIDF_PARAMS_WORD_NOSTOP.update({'train': train_cleaned, 'test': test_cleaned})
    post_train, post_test = run_tfidf(**TFIDF_PARAMS_WORD_NOSTOP)
    save_in_cache('tfidf_word_nostop', post_train, post_test)
    del post_train
    del post_test
    gc.collect()


print('~~~~~~~~~~~~~~~~~~~')
print_step('Run TFIDF CHAR')
if not is_in_cache('tfidf_char_cleaned'):
    TFIDF_PARAMS_CHAR.update({'train': train_cleaned, 'test': test_cleaned})
    post_train_cleaned, post_test_cleaned = run_tfidf(**TFIDF_PARAMS_CHAR)
    save_in_cache('tfidf_char_cleaned', post_train_cleaned, post_test_cleaned)
    del post_test_cleaned
    del post_train_cleaned
    gc.collect()
del train_cleaned
del test_cleaned
gc.collect()

if not is_in_cache('tfidf_char'):
    TFIDF_PARAMS_CHAR.update({'train': train, 'test': test})
    post_train, post_test = run_tfidf(**TFIDF_PARAMS_CHAR)
    save_in_cache('tfidf_char', post_train, post_test)
    del post_train
    del post_test
    gc.collect()


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run TFIDF WORD-CHAR UNION')
if is_in_cache('tfidf_char_union'):
    post_train, post_test = load_cache('tfidf_char_union')
else:
    TFIDF_UNION1.update({'train': train, 'test': test})
    post_trainw, post_testw = run_tfidf(**TFIDF_UNION1)
    TFIDF_UNION2.update({'train': train, 'test': test})
    post_trainc, post_testc = run_tfidf(**TFIDF_UNION2)
    post_train = csr_matrix(hstack([post_trainw, post_trainc]))
    del post_trainw; del post_trainc; gc.collect()
    post_test = csr_matrix(hstack([post_testw, post_testc]))
    del post_testw; del post_testc; gc.collect()
    save_in_cache('tfidf_char_union', post_train, post_test)


print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Word LR Sag')
train, test = run_cv_model(label='tfidf_word_lr_sag',
                           data_key='tfidf_word',
                           model_fn=runSagLR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)
# toxic CV scores : [0.9757770727127603, 0.9754469511129109, 0.9748022104865504, 0.9727014869411932, 0.9753668774625703]
# toxic mean CV : 0.9748189197431969
# severe_toxic CV scores : [0.9822782217978469, 0.9809759688772627, 0.982837995178992, 0.9888689680969123, 0.9832500976058173]
# severe_toxic mean CV : 0.9836422503113663
# obscene CV scores : [0.9863031895889313, 0.9859709183099142, 0.986069037576627, 0.984788656923766, 0.9868893669717265]
# obscene mean CV : 0.986004233874193
# threat CV scores : [0.9892926265229371, 0.9904405583142148, 0.986893640592099, 0.9938866116828939, 0.982301808641914]
# threat mean CV : 0.9885630491508117
# insult CV scores : [0.9778670021983397, 0.9786535248142688, 0.9773924913032992, 0.9796894980895773, 0.9802901280493739]
# insult mean CV : 0.9787785288909717
# identity_hate CV scores : [0.9733299704336319, 0.9725159196222063, 0.9727119512226128, 0.9780407482478374, 0.9787762183124902]
# identity_hate mean CV : 0.9750749615677557
# ('tfidf_word_lr overall : ', 0.9811469905897159)

print('~~~~~~~~~~~~~~~~~~~')
print_step('Run Word LR L1')
train, test = run_cv_model(label='tfidf_word_lr_l1',
                           data_key='tfidf_word',
                           model_fn=runL1LR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)

print('~~~~~~~~~~~~~~~~~~~')
print_step('Run Word LR L2')
train, test = run_cv_model(label='tfidf_word_lr_l2',
                           data_key='tfidf_word',
                           model_fn=runL2LR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)

print('~~~~~~~~~~~~~~~~~~')
print_step('Run Word NBLR')
train, test = run_cv_model(label='tfidf_word_nblr',
                           data_key='tfidf_word',
                           model_fn=runNBLR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)
# toxic CV scores : [0.9772131130009183, 0.9775146406777059, 0.9769621008062486, 0.9753843107161422, 0.9772501917811696]
# toxic mean CV : 0.976864871396437
# severe_toxic CV scores : [0.975601748723401, 0.9808998685856111, 0.9763889171384901, 0.9828484130807174, 0.9684032978115956]
# severe_toxic mean CV : 0.976828449067963
# obscene CV scores : [0.9879042584952109, 0.9869183882224537, 0.9873306628064691, 0.9865324805554424, 0.987983036312878]
# obscene mean CV : 0.9873337652784908
# threat CV scores : [0.9806477000115235, 0.9848568923703866, 0.9898917182605781, 0.9936848060184404, 0.9628761938790026]
# threat mean CV : 0.9823914621079861
# insult CV scores : [0.9785862018078957, 0.9779854187444984, 0.9780877460359315, 0.9795533853696614, 0.9778310819790152]
# insult mean CV : 0.9784087667874004
# identity_hate CV scores : [0.9634842640818237, 0.9718723059717469, 0.9661756895390451, 0.976226682505195, 0.9699773525845178]
# identity_hate mean CV : 0.9695472589364658
# ('tfidf_word_nblr overall : ', 0.9785624289291238)


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run Word No-stop LR Sag')
train, test = run_cv_model(label='tfidf_word_nostop_lr_sag',
                           data_key='tfidf_word_nostop',
                           model_fn=runSagLR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)
# toxic CV scores : [0.9755463841013073, 0.9753979313406889, 0.9748562847831383, 0.9726387684610107, 0.9752481878960215]
# toxic mean CV : 0.9747375113164335
# severe_toxic CV scores : [0.982322620497575, 0.9808813147987286, 0.982854564603641, 0.9888985350941901, 0.9835449738337034]
# severe_toxic mean CV : 0.9837004017655676
# obscene CV scores : [0.9864381341124995, 0.9858908873781942, 0.9862271469673383, 0.9847086037664334, 0.9868721088185911]
# obscene mean CV : 0.9860273762086115
# threat CV scores : [0.9894762825146401, 0.990457909058529, 0.9871404794829085, 0.9938928974330982, 0.9825343813994726]
# threat mean CV : 0.9887003899777296
# insult CV scores : [0.977881202981604, 0.9786934501386447, 0.9775995065286077, 0.9797496019843493, 0.98026534902532]
# insult mean CV : 0.9788378221317051
# identity_hate CV scores : [0.9729595189823463, 0.9723255692819551, 0.9726186885559058, 0.9780060981859006, 0.9788728559852302]
# identity_hate mean CV : 0.9749565461982677
# ('tfidf_word_nostop_lr overall : ', 0.9811600079330525)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run Word No-stop L1 LR')
train, test = run_cv_model(label='tfidf_word_nostop_lr_l1',
                           data_key='tfidf_word_nostop',
                           model_fn=runL1LR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run Word No-stop L2 LR')
train, test = run_cv_model(label='tfidf_word_nostop_lr_l2',
                           data_key='tfidf_word_nostop',
                           model_fn=runL2LR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)


print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run Word No-stop NBLR')
train, test = run_cv_model(label='tfidf_word_nostop_nblr',
                           data_key='tfidf_word_nostop',
                           model_fn=runNBLR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)
# toxic CV scores : [0.9768566292540937, 0.9774012279345319, 0.9770977227221203, 0.9753824074096629, 0.9772751581819247]
# toxic mean CV : 0.9768026291004668
# severe_toxic CV scores : [0.9764123846477135, 0.9807363571375772, 0.9765748518797615, 0.9828629981431329, 0.9690244031906561]
# severe_toxic mean CV : 0.977122198999768
# obscene CV scores : [0.9878519290723909, 0.986992878852394, 0.9873912557440931, 0.9864634887906003, 0.9878080257111613]
# obscene mean CV : 0.987301515634128
# threat CV scores : [0.9813947641346366, 0.9846801112019024, 0.9902835831840515, 0.9935207148552128, 0.9630260593970312]
# threat mean CV : 0.9825810465545668
# insult CV scores : [0.9787361570597426, 0.9782284633723526, 0.9783654963463133, 0.9793892506737358, 0.9777182788205268]
# insult mean CV : 0.9784875292545342
# identity_hate CV scores : [0.962914918562554, 0.9721121564004795, 0.9658153513949407, 0.9754071185402243, 0.9695557018308171]
# identity_hate mean CV : 0.969161049345803
# ('tfidf_word_nostop_nblr overall : ', 0.9785759948148778)


print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Char Sag LR')
train, test = run_cv_model(label='tfidf_char_lr_sag',
                           data_key='tfidf_char',
                           model_fn=runSagLR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)
# toxic CV scores : [0.9784009673265261, 0.9812385913192474, 0.9795732673580841, 0.9770499644574512, 0.9796899916465122]
# toxic mean CV : 0.9791905564215643
# severe_toxic CV scores : [0.9874673632351383, 0.9849230141867016, 0.9863084958980751, 0.9911504389657307, 0.9863854891527318]
# severe_toxic mean CV : 0.9872469602876756
# obscene CV scores : [0.9905731863097772, 0.991212866029434, 0.9913262139731425, 0.9912080136949828, 0.9910154759990609]
# obscene mean CV : 0.9910671512012795
# threat CV scores : [0.992570607708183, 0.9912702840022211, 0.9888228469363168, 0.9921282557704841, 0.9825194940963572]
# threat mean CV : 0.9894622977027124
# insult CV scores : [0.9802835718646636, 0.9821645795907262, 0.981048710418828, 0.9833333804202983, 0.9831259868968353]
# insult mean CV : 0.9819912458382702
# identity_hate CV scores : [0.9822133804859271, 0.9835197330415227, 0.9850051856967694, 0.9856928431759572, 0.9845490536314333]
# identity_hate mean CV : 0.9841960392063219
# ('tfidf_char_lr overall : ', 0.9855257084429706)

print('~~~~~~~~~~~~~~~~~~~')
print_step('Run Char L1 LR')
train, test = run_cv_model(label='tfidf_char_lr_l1',
                           data_key='tfidf_char',
                           model_fn=runL1LR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)

print('~~~~~~~~~~~~~~~~~~~')
print_step('Run Char L2 LR')
train, test = run_cv_model(label='tfidf_char_lr_l2',
                           data_key='tfidf_char',
                           model_fn=runL2LR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)

print('~~~~~~~~~~~~~~~~~~')
print_step('Run Char NBLR')
train, test = run_cv_model(label='tfidf_char_nblr',
                           data_key='tfidf_char_cleaned',
                           model_fn=runNBLR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)
# toxic CV scores : [0.9807326522118874, 0.9826694203535984, 0.9809316730534118, 0.9790661098211237, 0.9824618401087325]
# toxic mean CV : 0.9811723391097507
# severe_toxic CV scores : [0.9832785071401046, 0.9865382354356493, 0.985616201523891, 0.9905662427242118, 0.9857921648449394]
# severe_toxic mean CV : 0.9863582703337592
# obscene CV scores : [0.9931372007772084, 0.9922604529147763, 0.9917650282231919, 0.9921780390833258, 0.9919211568424091]
# obscene mean CV : 0.9922523755681822
# threat CV scores : [0.9902325130687115, 0.9897313057188052, 0.992476979163393, 0.9941582222575106, 0.9778594373922739]
# threat mean CV : 0.9888916915201389
# insult CV scores : [0.9816600962424099, 0.9835163833118921, 0.983062159502385, 0.9851152453466306, 0.9855841225891987]
# insult mean CV : 0.9837876013985033
# identity_hate CV scores : [0.9784744419997673, 0.9832166574997753, 0.9821686056263825, 0.9854686302751767, 0.9827221066157655]
# identity_hate mean CV : 0.9824100884033735
# ('tfidf_char_nblr overall : ', 0.9858120610556179)


print('~~~~~~~~~~~~~~~~~~~~~')
print_step('Run Union Sag LR')
train, test = run_cv_model(label='tfidf_union_lr_sag',
                           data_key='tfidf_char_union',
                           model_fn=runSagLR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)
# toxic CV scores : [0.9790060562019675, 0.9809792294830445, 0.9788201584400681, 0.9767082416399386, 0.9801075300500608]
# toxic mean CV : 0.9791242431630159
# severe_toxic CV scores : [0.9887540822000006, 0.985225679036354, 0.9888101813657472, 0.9922708594917751, 0.9878202822656059]
# severe_toxic mean CV : 0.9885762168718966
# obscene CV scores : [0.9906413144023375, 0.9908903147498299, 0.9907209698159072, 0.9903223974990683, 0.9904509522112532]
# obscene mean CV : 0.9906051897356791
# threat CV scores : [0.992013092282389, 0.9919672601275967, 0.990500303801712, 0.9898749797367263, 0.9868821355670905]
# threat mean CV : 0.9902475543031029
# insult CV scores : [0.9809979089085217, 0.982573861222668, 0.982638617484855, 0.9846358267974009, 0.9840449559303801]
# insult mean CV : 0.9829782340687652
# identity_hate CV scores : [0.9811407249778776, 0.9853436988018616, 0.9797503013036635, 0.985873293498512, 0.986407444453307]
# identity_hate mean CV : 0.9837030926070444
# ('tfidf_union_lr overall : ', 0.985872421791584)

print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Union L1 LR')
train, test = run_cv_model(label='tfidf_union_lr_l1',
                           data_key='tfidf_char_union',
                           model_fn=runL1LR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)

print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Union L2 LR')
train, test = run_cv_model(label='tfidf_union_lr_l2',
                           data_key='tfidf_char_union',
                           model_fn=runL2LR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)

print('~~~~~~~~~~~~~~~~~~~')
print_step('Run Union NBLR')
train, test = run_cv_model(label='tfidf_union_nblr',
                           data_key='tfidf_char_union',
                           model_fn=runNBLR,
                           train=train,
                           test=test,
                           train_key='extra_label',
                           targets=['toxic', 'severe_toxic', 'obscene', 'insult',
                                    'threat', 'identity_hate', 'non_toxic'],
                           kf=kf)
# toxic CV scores : [0.9791019659296382, 0.9816840402316044, 0.9790436496717447, 0.9770628344345982, 0.9804720757663966]
# toxic mean CV : 0.9794729132067964
# severe_toxic CV scores : [0.9858194025591905, 0.9828453373192555, 0.9847209068932279, 0.9900745673718263, 0.9849418656279192]
# severe_toxic mean CV : 0.985680415954284
# obscene CV scores : [0.991036476916225, 0.989223251648142, 0.9897582371155332, 0.9897799389802687, 0.9898235468439166]
# obscene mean CV : 0.9899242903008171
# threat CV scores : [0.98767769781158, 0.9907193160271116, 0.9908774369611448, 0.9820263935342788, 0.9664302893760897]
# threat mean CV : 0.9835462267420411
# insult CV scores : [0.9786568920103006, 0.9812528119224003, 0.9812406368878238, 0.9839728423313907, 0.9829259234842991]
# insult mean CV : 0.981609821327243
# identity_hate CV scores : [0.9713743287606447, 0.9766728583027343, 0.9764916204787716, 0.9830147758889118, 0.9783922551261561]
# identity_hate mean CV : 0.9771891677114437
# ('tfidf_union_nblr overall : ', 0.982903805873771)

import pdb
pdb.set_trace()
print('~~~~~~~~~~~~~~~~~~')
print_step('Cache Level 2')
save_in_cache('lvl1_lr', train, test)
print_step('Done!')
