#
# -*- coding: UTF-8 -*-
# Copyright @2017. DataRobot, Inc. All Rights Reserved. Permission to use, copy, modify,
# and distribute this software and its documentation is hereby granted, provided that the
# above copyright notice, this paragraph and the following two paragraphs appear in all copies,
# modifications, and distributions of this software or its documentation. Contact DataRobot,
# 1 International Place, 5th Floor, Boston, MA, United States 02110, support@datarobot.com
# for more details.
#
# IN NO EVENT SHALL DATAROBOT BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS OR LOST DATA, ARISING OUT OF THE USE OF THIS
# SOFTWARE AND ITS DOCUMENTATION BASED ON ANY THEORY OF LIABILITY, EVEN IF DATAROBOT HAS BEEN
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS".
# DATAROBOT SPECIFICALLY DISCLAIMS ANY AND ALL WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  DATAROBOT HAS NO
# OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
#
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import re
from datetime import datetime
import sys
import os
import re
import calendar

def predict(row):
    round_tfidf_word_nostop_nblr_threat = np.float32(row[u'tfidf_word_nostop_nblr_threat'])
    round_tfidf_char_lr_threat = np.float32(row[u'tfidf_char_lr_threat'])
    round_num_lowercase = np.float32(row[u'num_lowercase'])
    round_tfidf_word_stop_lr_threat = np.float32(row[u'tfidf_word_stop_lr_threat'])
    round_num_unique_words = np.float32(row[u'num_unique_words'])
    round_tfidf_char_lr_insult = np.float32(row[u'tfidf_char_lr_insult'])
    round_tfidf_char_lr_severe_toxic = np.float32(row[u'tfidf_char_lr_severe_toxic'])
    round_tfidf_char_nblr_identity_hate = np.float32(row[u'tfidf_char_nblr_identity_hate'])
    round_num_words_lower = np.float32(row[u'num_words_lower'])
    round_tfidf_word_nostop_nblr_identity_hate = np.float32(row[u'tfidf_word_nostop_nblr_identity_hate'])
    round_tfidf_union_lr_obscene = np.float32(row[u'tfidf_union_lr_obscene'])
    comment_text_words = bag_of_words(row[u'comment_text'])
    round_num_stopwords = np.float32(row[u'num_stopwords'])
    round_lowercase_per_char = np.float32(row[u'lowercase_per_char'])
    round_tfidf_union_lr_toxic = np.float32(row[u'tfidf_union_lr_toxic'])
    round_tfidf_word_nostop_nblr_toxic = np.float32(row[u'tfidf_word_nostop_nblr_toxic'])
    round_num_words_title = np.float32(row[u'num_words_title'])
    round_tfidf_word_stop_lr_insult = np.float32(row[u'tfidf_word_stop_lr_insult'])
    round_tfidf_word_nostop_nblr_insult = np.float32(row[u'tfidf_word_nostop_nblr_insult'])
    round_tfidf_word_stop_nblr_insult = np.float32(row[u'tfidf_word_stop_nblr_insult'])
    round_tfidf_union_nblr_severe_toxic = np.float32(row[u'tfidf_union_nblr_severe_toxic'])
    round_tfidf_union_nblr_insult = np.float32(row[u'tfidf_union_nblr_insult'])
    round_tfidf_word_stop_nblr_severe_toxic = np.float32(row[u'tfidf_word_stop_nblr_severe_toxic'])
    round_tfidf_union_lr_severe_toxic = np.float32(row[u'tfidf_union_lr_severe_toxic'])
    round_tfidf_char_nblr_obscene = np.float32(row[u'tfidf_char_nblr_obscene'])
    round_tfidf_word_stop_lr_severe_toxic = np.float32(row[u'tfidf_word_stop_lr_severe_toxic'])
    round_tfidf_word_nostop_lr_severe_toxic = np.float32(row[u'tfidf_word_nostop_lr_severe_toxic'])
    round_tfidf_word_nostop_lr_obscene = np.float32(row[u'tfidf_word_nostop_lr_obscene'])
    round_num_chars = np.float32(row[u'num_chars'])
    round_tfidf_char_nblr_toxic = np.float32(row[u'tfidf_char_nblr_toxic'])
    round_tfidf_word_nostop_lr_toxic = np.float32(row[u'tfidf_word_nostop_lr_toxic'])
    round_tfidf_union_lr_insult = np.float32(row[u'tfidf_union_lr_insult'])
    round_num_capital = np.float32(row[u'num_capital'])
    round_num_words = np.float32(row[u'num_words'])
    round_tfidf_word_stop_nblr_obscene = np.float32(row[u'tfidf_word_stop_nblr_obscene'])
    round_tfidf_word_nostop_lr_insult = np.float32(row[u'tfidf_word_nostop_lr_insult'])
    round_tfidf_char_nblr_severe_toxic = np.float32(row[u'tfidf_char_nblr_severe_toxic'])
    round_tfidf_word_stop_lr_identity_hate = np.float32(row[u'tfidf_word_stop_lr_identity_hate'])
    round_tfidf_word_stop_nblr_threat = np.float32(row[u'tfidf_word_stop_nblr_threat'])
    round_tfidf_word_nostop_lr_threat = np.float32(row[u'tfidf_word_nostop_lr_threat'])
    round_punctuation_per_char = np.float32(row[u'punctuation_per_char'])
    round_tfidf_char_nblr_threat = np.float32(row[u'tfidf_char_nblr_threat'])
    round_tfidf_word_stop_nblr_identity_hate = np.float32(row[u'tfidf_word_stop_nblr_identity_hate'])
    round_tfidf_char_lr_toxic = np.float32(row[u'tfidf_char_lr_toxic'])
    round_tfidf_char_lr_obscene = np.float32(row[u'tfidf_char_lr_obscene'])
    round_num_punctuations = np.float32(row[u'num_punctuations'])
    round_tfidf_word_nostop_nblr_severe_toxic = np.float32(row[u'tfidf_word_nostop_nblr_severe_toxic'])
    round_num_words_upper = np.float32(row[u'num_words_upper'])
    round_tfidf_union_nblr_threat = np.float32(row[u'tfidf_union_nblr_threat'])
    round_tfidf_union_lr_threat = np.float32(row[u'tfidf_union_lr_threat'])
    round_capital_per_char = np.float32(row[u'capital_per_char'])
    round_tfidf_union_nblr_toxic = np.float32(row[u'tfidf_union_nblr_toxic'])
    round_tfidf_word_stop_lr_toxic = np.float32(row[u'tfidf_word_stop_lr_toxic'])
    round_tfidf_union_nblr_obscene = np.float32(row[u'tfidf_union_nblr_obscene'])
    round_tfidf_word_nostop_lr_identity_hate = np.float32(row[u'tfidf_word_nostop_lr_identity_hate'])
    round_tfidf_char_nblr_insult = np.float32(row[u'tfidf_char_nblr_insult'])
    round_tfidf_word_stop_nblr_toxic = np.float32(row[u'tfidf_word_stop_nblr_toxic'])
    round_tfidf_char_lr_identity_hate = np.float32(row[u'tfidf_char_lr_identity_hate'])
    round_tfidf_word_stop_lr_obscene = np.float32(row[u'tfidf_word_stop_lr_obscene'])
    round_tfidf_word_nostop_nblr_obscene = np.float32(row[u'tfidf_word_nostop_nblr_obscene'])
    return sum([
        -1.0461679,
           0.043707079842917717638 * (int(u'101' in comment_text_words)),
          -0.035428572765959204238 * (int(u'104' in comment_text_words)),
           0.043702479944662850353 * (int(u'105' in comment_text_words)),
          -0.025829132039179795999 * (int(u'110' in comment_text_words)),
         -0.0077636795206384896784 * (int(u'114' in comment_text_words)),
          -0.078281732614910767842 * (int(u'117' in comment_text_words)),
           0.068911075864093743193 * (int(u'118' in comment_text_words)),
          -0.015007437344343841842 * (int(u'129' in comment_text_words)),
           0.019584202219934201267 * (int(u'138' in comment_text_words)),
          -0.072176704197317795297 * (int(u'139' in comment_text_words)),
          -0.059050867797224272648 * (int(u'140' in comment_text_words)),
           0.023556460662557791169 * (int(u'146' in comment_text_words)),
          -0.055724804603036123818 * (int(u'156' in comment_text_words)),
            0.01187112239222144934 * (int(u'161' in comment_text_words)),
          -0.015860310026689708462 * (int(u'168' in comment_text_words)),
         0.00083653748445158804779 * (int(u'179' in comment_text_words)),
          -0.034940026751030646424 * (int(u'180' in comment_text_words)),
          -0.024188352834946896114 * (int(u'192' in comment_text_words)),
            0.29755099710972288118 * (int(u'1971' in comment_text_words)),
          -0.015836539874808111539 * (int(u'20' in comment_text_words)),
         -0.0014184575958181597048 * (int(u'2001' in comment_text_words)),
           0.014229027878372909124 * (int(u'205' in comment_text_words)),
          0.0057512542123961457249 * (int(u'207' in comment_text_words)),
         0.00079962811301944572602 * (int(u'216' in comment_text_words)),
          -0.053191840931695481831 * (int(u'223' in comment_text_words)),
           -0.01482337513136008228 * (int(u'224' in comment_text_words)),
          -0.041096101082020018713 * (int(u'227' in comment_text_words)),
          -0.061598648827470181555 * (int(u'241' in comment_text_words)),
          -0.060941180763319287295 * (int(u'245' in comment_text_words)),
         -0.0026681968540291528015 * (int(u'246' in comment_text_words)),
          -0.066738893537766075248 * (int(u'254' in comment_text_words)),
           0.053322616312954894402 * (int(u'27' in comment_text_words)),
          -0.012247327909803753729 * (int(u'28' in comment_text_words)),
           0.026851286489626050025 * (int(u'46' in comment_text_words)),
          -0.011966395593410778311 * (int(u'58' in comment_text_words)),
          -0.019812105201700128815 * (int(u'69' in comment_text_words)),
          0.0030209242620968710345 * (int(u'71' in comment_text_words)),
           0.058027668899718894135 * (int(u'78' in comment_text_words)),
           0.067577232367136183533 * (int(u'83' in comment_text_words)),
          -0.033462498450546586393 * (int(u'8th' in comment_text_words)),
           -0.01119127951978620035 * (int(u'92' in comment_text_words)),
         -0.0062523896980041142671 * (int(u'94' in comment_text_words)),
           0.017846500564348322798 * (int(u'99' in comment_text_words)),
           0.035183005947637568944 * (int(u'abilities' in comment_text_words)),
          -0.016567215185333956234 * (int(u'abu' in comment_text_words)),
          -0.028678475450985993489 * (int(u'abuse' in comment_text_words)),
          -0.028707030490027433262 * (int(u'abusive' in comment_text_words)),
           0.035490664128285366596 * (int(u'accent' in comment_text_words)),
          -0.014880307093507126256 * (int(u'accepted' in comment_text_words)),
          -0.016428547958256427275 * (int(u'according' in comment_text_words)),
          -0.049733194753807045962 * (int(u'account' in comment_text_words)),
        -0.00070293635500981649974 * (int(u'accusations' in comment_text_words)),
          -0.078616144161071863095 * (int(u'acquired' in comment_text_words)),
          -0.057888955920836301927 * (int(u'acting' in comment_text_words)),
          -0.016160630296937393557 * (int(u'activities' in comment_text_words)),
          -0.041889114088578843775 * (int(u'addressed' in comment_text_words)),
           -0.05204888236805083751 * (int(u'adequate' in comment_text_words)),
           0.040293453541962231945 * (int(u'administrator' in comment_text_words)),
         -0.0072821416559294095336 * (int(u'advice' in comment_text_words)),
            -0.0588365779393985322 * (int(u'advise' in comment_text_words)),
         -0.0039658949871502853154 * (int(u'advocated' in comment_text_words)),
          -0.040234593335524627966 * (int(u'affair' in comment_text_words)),
          -0.016738474284908690104 * (int(u'against' in comment_text_words)),
            0.18593629533876449011 * (int(u'aggression' in comment_text_words)),
         -0.0019934569650213907095 * (int(u'agreeing' in comment_text_words)),
          -0.025844103307663904984 * (int(u'agrees' in comment_text_words)),
           0.049275953428944967161 * (int(u'ahh' in comment_text_words)),
           0.057583289053191995255 * (int(u'aids' in comment_text_words)),
           0.062104506852328635047 * (int(u'aint' in comment_text_words)),
          -0.017785152814468838112 * (int(u'al' in comment_text_words)),
           0.014275726535753211627 * (int(u'alert' in comment_text_words)),
           -0.06605763014906702868 * (int(u'alex' in comment_text_words)),
          -0.010661151492199559543 * (int(u'all' in comment_text_words)),
          -0.017035701404247902829 * (int(u'allowed' in comment_text_words)),
           0.020221029235807239793 * (int(u'ally' in comment_text_words)),
          -0.023223092363767836033 * (int(u'along' in comment_text_words)),
         -0.0077066973027674111688 * (int(u'alphabet' in comment_text_words)),
          -0.024136848761898553606 * (int(u'always' in comment_text_words)),
         -0.0091816072203326874746 * (int(u'america' in comment_text_words)),
           -0.04074358039909986573 * (int(u'amusement' in comment_text_words)),
           0.033482755440023492799 * (int(u'anal' in comment_text_words)),
            0.05709462901672320706 * (int(u'anarchism' in comment_text_words)),
          -0.025940813673988406818 * (int(u'ancestors' in comment_text_words)),
         -0.0079956501017658717506 * (int(u'andy' in comment_text_words)),
          -0.075660539921789421447 * (int(u'anglo' in comment_text_words)),
           0.011726441006727846353 * (int(u'angry' in comment_text_words)),
         -0.0027448197464935127424 * (int(u'annoying' in comment_text_words)),
          -0.025826112278736729905 * (int(u'anonymously' in comment_text_words)),
          -0.058375341465965936705 * (int(u'answers' in comment_text_words)),
          -0.014452434112826312643 * (int(u'apologize' in comment_text_words)),
           0.020144117626444732472 * (int(u'appealing' in comment_text_words)),
          -0.025138290015433127633 * (int(u'apply' in comment_text_words)),
          -0.053690410011412163793 * (int(u'arabic' in comment_text_words)),
         -0.0086147180608870266988 * (int(u'are' in comment_text_words)),
         -0.0089352550601698756128 * (int(u'around' in comment_text_words)),
            0.04160835965198086267 * (int(u'arsehole' in comment_text_words)),
             0.2976482930067784749 * (int(u'articles_for_deletion' in comment_text_words)),
           -0.03132719476508548373 * (int(u'asian' in comment_text_words)),
          -0.025525886345314170722 * (int(u'asians' in comment_text_words)),
           0.085600556839022776123 * (int(u'assassination' in comment_text_words)),
          -0.060588405009989333705 * (int(u'association' in comment_text_words)),
          -0.049006821282152251973 * (int(u'assuming' in comment_text_words)),
          0.0078090347874474868445 * (int(u'attack' in comment_text_words)),
          -0.019687206011256456334 * (int(u'attempted' in comment_text_words)),
            0.05336669730614697349 * (int(u'attitudes' in comment_text_words)),
            0.12125663724952430444 * (int(u'aussie' in comment_text_words)),
          -0.015654681429842085688 * (int(u'australians' in comment_text_words)),
           -0.02184931193661988208 * (int(u'avoid' in comment_text_words)),
           0.021854500082106674552 * (int(u'awful' in comment_text_words)),
         -0.0084090350080038168273 * (int(u'backup' in comment_text_words)),
          -0.067143138027977722349 * (int(u'badge' in comment_text_words)),
          0.0093376822374026963863 * (int(u'bald' in comment_text_words)),
          -0.035710944773393073215 * (int(u'balls' in comment_text_words)),
             0.0671897115750988011 * (int(u'bang' in comment_text_words)),
           -0.01466019189686665683 * (int(u'banks' in comment_text_words)),
         -0.0016923652713724257612 * (int(u'banned' in comment_text_words)),
           0.069804632558779389218 * (int(u'bans' in comment_text_words)),
           0.032609242969873157569 * (int(u'bare' in comment_text_words)),
         -0.0071401685152706574275 * (int(u'basement' in comment_text_words)),
          -0.095349586813273656816 * (int(u'bash' in comment_text_words)),
          -0.038620532842331717349 * (int(u'bastard' in comment_text_words)),
           -0.10152206402357707993 * (int(u'bat' in comment_text_words)),
         0.00085829772732186945111 * (int(u'be' in comment_text_words)),
          -0.048908674163098325716 * (int(u'beat' in comment_text_words)),
         -0.0040717092332154562404 * (int(u'because' in comment_text_words)),
           0.021527458780808337441 * (int(u'becuase' in comment_text_words)),
         -0.0012065149812718874243 * (int(u'bee' in comment_text_words)),
            0.15859139968240221208 * (int(u'behaviors' in comment_text_words)),
          -0.012515769949046911744 * (int(u'behind' in comment_text_words)),
           -0.19376207120627628178 * (int(u'believer' in comment_text_words)),
           0.049076810444922853993 * (int(u'believers' in comment_text_words)),
           0.089269268439581633823 * (int(u'belive' in comment_text_words)),
           0.017046945373593316853 * (int(u'beloved' in comment_text_words)),
          -0.069798131358247597622 * (int(u'bible' in comment_text_words)),
           -0.12341183358392615599 * (int(u'bigoted' in comment_text_words)),
           0.037562973772496767444 * (int(u'bitching' in comment_text_words)),
          -0.033219350105755933444 * (int(u'blacks' in comment_text_words)),
          -0.011304310332549810311 * (int(u'blanked' in comment_text_words)),
         -0.0051445118523745147696 * (int(u'blatantly' in comment_text_words)),
           0.039559989199721562469 * (int(u'blessed' in comment_text_words)),
           -0.10219213916919360485 * (int(u'blew' in comment_text_words)),
           0.041416091987701061927 * (int(u'blind' in comment_text_words)),
          -0.026962203300081158397 * (int(u'blood' in comment_text_words)),
           0.043636070488367838249 * (int(u'blue' in comment_text_words)),
           0.032033565474856091904 * (int(u'bnp' in comment_text_words)),
            0.12720604831349718866 * (int(u'bollocks' in comment_text_words)),
          -0.033609282662810761877 * (int(u'bomb' in comment_text_words)),
            0.11964990336720023045 * (int(u'bombs' in comment_text_words)),
           0.041002667975921253118 * (int(u'boo' in comment_text_words)),
          -0.039316877129591448758 * (int(u'borderline' in comment_text_words)),
           0.046639539738781182754 * (int(u'bored' in comment_text_words)),
            -0.1027426601434939385 * (int(u'bothering' in comment_text_words)),
          -0.094801861666099440651 * (int(u'bothers' in comment_text_words)),
          -0.030466561463523052916 * (int(u'boyfriend' in comment_text_words)),
          -0.012707226069739622348 * (int(u'boys' in comment_text_words)),
            0.12283616003028006991 * (int(u'brad' in comment_text_words)),
           0.035305172419849296939 * (int(u'brittanica' in comment_text_words)),
            0.03380168037618890009 * (int(u'bro' in comment_text_words)),
          -0.014451633534118078078 * (int(u'broke' in comment_text_words)),
           0.066316108475481763684 * (int(u'broken' in comment_text_words)),
          -0.018763970153941891827 * (int(u'brown' in comment_text_words)),
           -0.10032022096679463208 * (int(u'browse' in comment_text_words)),
           0.010586910431614257946 * (int(u'bs' in comment_text_words)),
          -0.032140082885980129912 * (int(u'buck' in comment_text_words)),
           0.020870348457603184883 * (int(u'bull' in comment_text_words)),
            0.05461130993022478286 * (int(u'bullies' in comment_text_words)),
          -0.055448624213135754013 * (int(u'bully' in comment_text_words)),
           0.016433644204197333377 * (int(u'bullying' in comment_text_words)),
          -0.018586954120423760523 * (int(u'bunch' in comment_text_words)),
          -0.052158363806869993684 * (int(u'bureaucracy' in comment_text_words)),
          -0.016613942643997393206 * (int(u'burn' in comment_text_words)),
           0.074172851161243599782 * (int(u'burned' in comment_text_words)),
            0.02003591130637585177 * (int(u'butt' in comment_text_words)),
          -0.096971531641123140388 * (int(u'buzz' in comment_text_words)),
           -0.14302433948041262668 * (int(u'cabal' in comment_text_words)),
           -0.11761389635773202011 * (int(u'calendar' in comment_text_words)),
          -0.017338209389857420911 * (int(u'calls' in comment_text_words)),
           0.067325260883225421238 * (int(u'cancer' in comment_text_words)),
           -0.02736518961783122178 * (int(u'cant' in comment_text_words)),
           0.027831533878822679873 * (int(u'capital' in comment_text_words)),
          -0.011441336326818879118 * (int(u'careful' in comment_text_words)),
          -0.038354224081341418973 * (int(u'careless' in comment_text_words)),
          -0.032305140765611167764 * (int(u'catholic' in comment_text_words)),
          -0.017111635059060444075 * (int(u'caused' in comment_text_words)),
          0.0071059714105037143431 * (int(u'celebrities' in comment_text_words)),
          0.0063256076242637409787 * (int(u'censor' in comment_text_words)),
          -0.020605288324647679649 * (int(u'censoring' in comment_text_words)),
          0.0033375478063674057436 * (int(u'certificate' in comment_text_words)),
           -0.10301965881813349157 * (int(u'ch' in comment_text_words)),
           0.033064670516825149515 * (int(u'champions' in comment_text_words)),
           -0.01077617093005204911 * (int(u'chance' in comment_text_words)),
           -0.02505101136120981356 * (int(u'character' in comment_text_words)),
          -0.028233563518016497468 * (int(u'cheap' in comment_text_words)),
          -0.027840029022571041034 * (int(u'chest' in comment_text_words)),
            -0.1000612347961833265 * (int(u'chips' in comment_text_words)),
           0.057852287306009983481 * (int(u'choosing' in comment_text_words)),
          -0.010690564288427155046 * (int(u'christian' in comment_text_words)),
          -0.075810786534255131253 * (int(u'citizens' in comment_text_words)),
         -0.0087655714739497628957 * (int(u'city' in comment_text_words)),
           0.017152825152975333806 * (int(u'civil' in comment_text_words)),
           0.020353958860458688795 * (int(u'civilians' in comment_text_words)),
           0.057548027049403219224 * (int(u'ck' in comment_text_words)),
           -0.10690878946633999846 * (int(u'clark' in comment_text_words)),
            0.11041958298571176444 * (int(u'clay' in comment_text_words)),
          -0.051414897798724067124 * (int(u'cleansing' in comment_text_words)),
         -0.0099409457441904172659 * (int(u'clown' in comment_text_words)),
           0.060770577206119828773 * (int(u'clowns' in comment_text_words)),
            0.11276962374204420325 * (int(u'cn' in comment_text_words)),
           0.026160982634094945981 * (int(u'cole' in comment_text_words)),
            0.05492864838294000962 * (int(u'collins' in comment_text_words)),
         -0.0057559743338104060392 * (int(u'communism' in comment_text_words)),
          -0.013378525503444371747 * (int(u'compared' in comment_text_words)),
           0.020146515590221274888 * (int(u'competitors' in comment_text_words)),
           0.023511605103849874537 * (int(u'complete' in comment_text_words)),
          -0.037319869246718616329 * (int(u'comply' in comment_text_words)),
          -0.010395119057999538739 * (int(u'comprehension' in comment_text_words)),
         -0.0010600958611568383778 * (int(u'computer' in comment_text_words)),
          -0.064950031110748818186 * (int(u'condemned' in comment_text_words)),
           -0.17933021972484253226 * (int(u'conjunction' in comment_text_words)),
           0.031282088420623258007 * (int(u'connolley' in comment_text_words)),
          -0.017803422155637162183 * (int(u'considering' in comment_text_words)),
          0.0080750107681418735461 * (int(u'constructive' in comment_text_words)),
          -0.012366273180167962689 * (int(u'contain' in comment_text_words)),
          -0.080205507872144032877 * (int(u'contempt' in comment_text_words)),
         -0.0086506685751860205869 * (int(u'contribution' in comment_text_words)),
           -0.10422462554586987837 * (int(u'convert' in comment_text_words)),
          -0.033138988273575295529 * (int(u'convicted' in comment_text_words)),
          -0.045001209131539356145 * (int(u'cos' in comment_text_words)),
         -0.0099986753841123481024 * (int(u'could' in comment_text_words)),
           0.044634340340100095434 * (int(u'coulter' in comment_text_words)),
           -0.05057172373306540597 * (int(u'counts' in comment_text_words)),
           0.023323955990306545483 * (int(u'coward' in comment_text_words)),
        -0.00025800544746459576568 * (int(u'credible' in comment_text_words)),
           0.037298024359736899058 * (int(u'creep' in comment_text_words)),
         -0.0067518792809415405579 * (int(u'criticised' in comment_text_words)),
          -0.093704899634701036004 * (int(u'criticisms' in comment_text_words)),
          -0.051048498527290590843 * (int(u'critics' in comment_text_words)),
            0.12718657196007165933 * (int(u'cruel' in comment_text_words)),
           0.052486355723721837829 * (int(u'cruft' in comment_text_words)),
           -0.02792268516639133738 * (int(u'crying' in comment_text_words)),
           0.019291284722910313765 * (int(u'csd' in comment_text_words)),
          -0.020306130429551993294 * (int(u'cult' in comment_text_words)),
          -0.038979662717347145218 * (int(u'cultural' in comment_text_words)),
           0.053474231790202692016 * (int(u'cum' in comment_text_words)),
         -0.0079008823112440614705 * (int(u'curse' in comment_text_words)),
           0.039595921045728767196 * (int(u'customer' in comment_text_words)),
           0.041908353487871627396 * (int(u'damn' in comment_text_words)),
         -0.0098675257377502899325 * (int(u'dare' in comment_text_words)),
         -0.0042853487234597309463 * (int(u'dave' in comment_text_words)),
            0.11080860289062667567 * (int(u'dawn' in comment_text_words)),
           -0.15660857801676400514 * (int(u'debt' in comment_text_words)),
          -0.098123333013976785089 * (int(u'defence' in comment_text_words)),
          -0.012012737428360651532 * (int(u'defending' in comment_text_words)),
          -0.026460646592102444324 * (int(u'degrees' in comment_text_words)),
         -0.0068394536480708995746 * (int(u'delete' in comment_text_words)),
           0.037843414599055449909 * (int(u'demand' in comment_text_words)),
           -0.07380303648170387365 * (int(u'democracy' in comment_text_words)),
          -0.015809465501267608994 * (int(u'democratic' in comment_text_words)),
         -0.0052942519351410737699 * (int(u'democrats' in comment_text_words)),
           0.028243905632696169467 * (int(u'demonstrated' in comment_text_words)),
           0.030195825774618448301 * (int(u'demonstrating' in comment_text_words)),
             -0.086960273916302816 * (int(u'dennis' in comment_text_words)),
           -0.10624323716675357354 * (int(u'desert' in comment_text_words)),
           0.096905549444396679726 * (int(u'desperate' in comment_text_words)),
            0.06061921967230241709 * (int(u'desperately' in comment_text_words)),
          -0.058039271373997759762 * (int(u'despicable' in comment_text_words)),
          -0.010460917902023480347 * (int(u'despite' in comment_text_words)),
          0.0092427023225355505603 * (int(u'destroy' in comment_text_words)),
           0.094713790924162141738 * (int(u'detailing' in comment_text_words)),
           -0.10708423561104628285 * (int(u'detected' in comment_text_words)),
           -0.02057954291713693859 * (int(u'devoid' in comment_text_words)),
          -0.022062273931759707885 * (int(u'di' in comment_text_words)),
            0.11285867181492414968 * (int(u'diamond' in comment_text_words)),
          -0.012019960504005973781 * (int(u'dick' in comment_text_words)),
          -0.023506975978152232221 * (int(u'did' in comment_text_words)),
          -0.097672088510264168382 * (int(u'differ' in comment_text_words)),
          -0.096274845767435474064 * (int(u'differing' in comment_text_words)),
           0.019277798073155261865 * (int(u'digest' in comment_text_words)),
           0.072098969432442289174 * (int(u'digging' in comment_text_words)),
           0.043292157692408531067 * (int(u'dirt' in comment_text_words)),
           0.069092459590752400289 * (int(u'discredited' in comment_text_words)),
          -0.010683955305155128784 * (int(u'discrimination' in comment_text_words)),
          -0.016201053702062699624 * (int(u'disease' in comment_text_words)),
            0.04007509916065251504 * (int(u'disgrace' in comment_text_words)),
            0.16821885730314983798 * (int(u'disguised' in comment_text_words)),
          -0.079370730083177840064 * (int(u'disgusted' in comment_text_words)),
           0.060871179837279278113 * (int(u'disgusting' in comment_text_words)),
           0.079155825774776478188 * (int(u'dishonesty' in comment_text_words)),
           0.049891179510933948704 * (int(u'disrespect' in comment_text_words)),
           0.027012640023665952099 * (int(u'disrupting' in comment_text_words)),
             0.1964334081537084109 * (int(u'distortions' in comment_text_words)),
          -0.088712601252657188877 * (int(u'doctor' in comment_text_words)),
          -0.001046452786143725654 * (int(u'documentation' in comment_text_words)),
          -0.006700091026351728124 * (int(u'doesn' in comment_text_words)),
          -0.024530304667543557928 * (int(u'dog' in comment_text_words)),
           0.031612036446813276958 * (int(u'dogma' in comment_text_words)),
           -0.14464985473298486163 * (int(u'dominate' in comment_text_words)),
           0.024429817364514154476 * (int(u'donations' in comment_text_words)),
          -0.015670521160010341538 * (int(u'dont' in comment_text_words)),
           0.031375217070518851559 * (int(u'doom' in comment_text_words)),
           0.026888235495580618123 * (int(u'dos' in comment_text_words)),
           0.017306864764940772111 * (int(u'dose' in comment_text_words)),
           0.012591469075286177565 * (int(u'down' in comment_text_words)),
           0.049651859182786979574 * (int(u'dozen' in comment_text_words)),
           -0.11678445834936528047 * (int(u'drag' in comment_text_words)),
           -0.10504648487338377749 * (int(u'dragged' in comment_text_words)),
           0.050760262598640686071 * (int(u'dragon' in comment_text_words)),
          -0.054123495730041701335 * (int(u'dramatic' in comment_text_words)),
          0.0058951217469539176838 * (int(u'drawn' in comment_text_words)),
        -0.00052464878667782851966 * (int(u'driving' in comment_text_words)),
           0.034517948221754288229 * (int(u'dry' in comment_text_words)),
           -0.10318140708240244874 * (int(u'duke' in comment_text_words)),
           0.014744937585686292494 * (int(u'dumbass' in comment_text_words)),
            0.26156015965177314975 * (int(u'duplicate' in comment_text_words)),
            -0.1343634469537403342 * (int(u'dust' in comment_text_words)),
          -0.027673958730374693255 * (int(u'dynamic' in comment_text_words)),
          0.0099069713230302731966 * (int(u'eagle' in comment_text_words)),
           0.078567311550114010688 * (int(u'earn' in comment_text_words)),
          -0.076896299215526042747 * (int(u'earned' in comment_text_words)),
           0.024059082504553151727 * (int(u'easy' in comment_text_words)),
           0.041048446609868394219 * (int(u'eaten' in comment_text_words)),
          -0.017708846347148431838 * (int(u'ed' in comment_text_words)),
         -0.0088074281902906469133 * (int(u'edited' in comment_text_words)),
          -0.058504065861299914264 * (int(u'edward' in comment_text_words)),
            0.27164694784714604747 * (int(u'egg' in comment_text_words)),
         -0.0039911872436363803746 * (int(u'eh' in comment_text_words)),
          -0.066494613552584305727 * (int(u'elite' in comment_text_words)),
          -0.039313628207036849238 * (int(u'emo' in comment_text_words)),
          -0.013710634204290125804 * (int(u'ended' in comment_text_words)),
            0.17867988922134483976 * (int(u'endorsing' in comment_text_words)),
          -0.005831763719775361933 * (int(u'enemies' in comment_text_words)),
          -0.054570627175256451735 * (int(u'enforcement' in comment_text_words)),
           0.014449731477502950755 * (int(u'engineering' in comment_text_words)),
           0.020990775455849531445 * (int(u'ep' in comment_text_words)),
            0.30603208165230710858 * (int(u'epic' in comment_text_words)),
           0.086404697775177924379 * (int(u'erasing' in comment_text_words)),
          -0.062619744143457306551 * (int(u'escalate' in comment_text_words)),
            0.15229657670506785427 * (int(u'esteem' in comment_text_words)),
            0.20127722642566106548 * (int(u'evans' in comment_text_words)),
         -0.0054596888956079140873 * (int(u'everyone' in comment_text_words)),
           -0.11091141543383180312 * (int(u'ex' in comment_text_words)),
          -0.034917931113620750971 * (int(u'exceed' in comment_text_words)),
          -0.023104369702362678035 * (int(u'exception' in comment_text_words)),
          -0.065285027473840448464 * (int(u'expression' in comment_text_words)),
            0.16312491964027428515 * (int(u'extraordinary' in comment_text_words)),
          -0.072752181774549645743 * (int(u'extremists' in comment_text_words)),
          -0.085992644230933848459 * (int(u'eyed' in comment_text_words)),
           -0.19261946928029630155 * (int(u'fabricated' in comment_text_words)),
         0.00045893475148490531701 * (int(u'fact' in comment_text_words)),
          -0.087731026035129652807 * (int(u'faggot' in comment_text_words)),
          0.0030234901772100519594 * (int(u'fags' in comment_text_words)),
         -0.0091759099806735071325 * (int(u'failure' in comment_text_words)),
           0.030604978192989826941 * (int(u'fallacy' in comment_text_words)),
          -0.019017170128200734658 * (int(u'fan' in comment_text_words)),
          -0.029976259205638854793 * (int(u'far' in comment_text_words)),
         -0.0055354074467543557328 * (int(u'fat' in comment_text_words)),
          -0.033244430903033590774 * (int(u'favor' in comment_text_words)),
          -0.045857610705217898006 * (int(u'favour' in comment_text_words)),
            0.14784711703509184622 * (int(u'feces' in comment_text_words)),
           -0.11557456560292167502 * (int(u'fed' in comment_text_words)),
           0.061019576957123691785 * (int(u'feeding' in comment_text_words)),
            0.17537649924666434686 * (int(u'fenian' in comment_text_words)),
          -0.012658663063203066598 * (int(u'fight' in comment_text_words)),
            0.13510495743239650523 * (int(u'filth' in comment_text_words)),
           0.091503536429230383775 * (int(u'filthy' in comment_text_words)),
          -0.065547206544249975169 * (int(u'fingers' in comment_text_words)),
          -0.062204107438861504231 * (int(u'firefox' in comment_text_words)),
           0.091632377833306022374 * (int(u'fish' in comment_text_words)),
          -0.052879308299754262945 * (int(u'flaw' in comment_text_words)),
          -0.078648162134891300146 * (int(u'flew' in comment_text_words)),
          -0.048174138333224147956 * (int(u'floyd' in comment_text_words)),
           0.022620600743280282197 * (int(u'fo' in comment_text_words)),
           -0.04126500517255244499 * (int(u'focus' in comment_text_words)),
           0.054442608857767985087 * (int(u'followers' in comment_text_words)),
          -0.063039884076182994832 * (int(u'follows' in comment_text_words)),
          -0.028954126634792257122 * (int(u'food' in comment_text_words)),
          -0.073087926680770715082 * (int(u'foolish' in comment_text_words)),
          -0.054555802452752918952 * (int(u'forbid' in comment_text_words)),
          -0.080966957402291872548 * (int(u'forensic' in comment_text_words)),
         -0.0070785872415556260959 * (int(u'forget' in comment_text_words)),
          -0.076697942907558672165 * (int(u'forgetting' in comment_text_words)),
           0.052469016885176134046 * (int(u'fortune' in comment_text_words)),
           0.017885780651211831416 * (int(u'frank' in comment_text_words)),
         -0.0031315991192317023037 * (int(u'frankly' in comment_text_words)),
          -0.036777289006789011527 * (int(u'friendly' in comment_text_words)),
              0.014142741187961912 * (int(u'frustrated' in comment_text_words)),
           -0.10565530366213032876 * (int(u'frustration' in comment_text_words)),
           0.056590389160957425829 * (int(u'fu' in comment_text_words)),
           0.054563328193132101807 * (int(u'fuck' in comment_text_words)),
           -0.11009641922892755839 * (int(u'fucked' in comment_text_words)),
           0.032956690984284504886 * (int(u'fucking' in comment_text_words)),
           0.037764327618931163577 * (int(u'fundamental' in comment_text_words)),
           0.066344873592218661651 * (int(u'fundamentalist' in comment_text_words)),
          -0.023776830631748447492 * (int(u'future' in comment_text_words)),
          -0.088378593742449371162 * (int(u'fyrom' in comment_text_words)),
          -0.001080633204114127624 * (int(u'gaming' in comment_text_words)),
          -0.044102855245227223779 * (int(u'gandhi' in comment_text_words)),
           0.065280603590002039827 * (int(u'gee' in comment_text_words)),
           0.001705690367799841305 * (int(u'genre' in comment_text_words)),
          -0.093728356503319495974 * (int(u'gentlemen' in comment_text_words)),
          -0.019316061532125391315 * (int(u'german' in comment_text_words)),
         -0.0057130942048889557863 * (int(u'get' in comment_text_words)),
          -0.052581627516699001867 * (int(u'girlfriend' in comment_text_words)),
          -0.057203232219000130221 * (int(u'glaring' in comment_text_words)),
          -0.012211567996308927281 * (int(u'globe' in comment_text_words)),
           0.085117527531559114551 * (int(u'glorious' in comment_text_words)),
          -0.003610548697419669207 * (int(u'go' in comment_text_words)),
          0.0005918744965155120193 * (int(u'god' in comment_text_words)),
           0.022606163385023709395 * (int(u'goddamn' in comment_text_words)),
          -0.022630207756037353656 * (int(u'goodbye' in comment_text_words)),
          -0.046790704195691902589 * (int(u'gore' in comment_text_words)),
            0.01714357331438116569 * (int(u'gossip' in comment_text_words)),
          -0.046984804120139687933 * (int(u'grass' in comment_text_words)),
         -0.0046478003325991440811 * (int(u'great' in comment_text_words)),
            0.07279349820768694912 * (int(u'grey' in comment_text_words)),
          -0.010950485012877505739 * (int(u'grief' in comment_text_words)),
         -0.0052883160076500762559 * (int(u'group' in comment_text_words)),
          -0.018378259374005723181 * (int(u'groups' in comment_text_words)),
          -0.026708638778471995023 * (int(u'grow' in comment_text_words)),
          -0.056918035191370522363 * (int(u'guiding' in comment_text_words)),
          -0.033255470331269958162 * (int(u'guilt' in comment_text_words)),
          -0.048794898097233536938 * (int(u'guilty' in comment_text_words)),
          -0.083415166345591457153 * (int(u'gwen' in comment_text_words)),
           -0.06684096844735405718 * (int(u'hahaha' in comment_text_words)),
           0.016663301846624883201 * (int(u'hammer' in comment_text_words)),
          -0.045354533499432414523 * (int(u'hang' in comment_text_words)),
          0.0090512830305560353777 * (int(u'harder' in comment_text_words)),
           0.018107316563849465418 * (int(u'harrass' in comment_text_words)),
           -0.15135886382826446717 * (int(u'harrassed' in comment_text_words)),
          -0.071516419492034674632 * (int(u'hates' in comment_text_words)),
          0.0043296268160622642523 * (int(u'hating' in comment_text_words)),
          -0.037994847990193154896 * (int(u'havent' in comment_text_words)),
          -0.011719633420564356044 * (int(u'he' in comment_text_words)),
          -0.059053087449201492609 * (int(u'heavy' in comment_text_words)),
          -0.087313279578320232566 * (int(u'heck' in comment_text_words)),
          -0.099513349101640002914 * (int(u'hehe' in comment_text_words)),
          0.0055949894661204554666 * (int(u'hell' in comment_text_words)),
         -0.0079666300031974428969 * (int(u'hello' in comment_text_words)),
         -0.0028622008464573387571 * (int(u'helping' in comment_text_words)),
          0.0057117149407831194902 * (int(u'here' in comment_text_words)),
          -0.065757587680169052313 * (int(u'heroes' in comment_text_words)),
           -0.02155815944208971241 * (int(u'hers' in comment_text_words)),
          -0.032784485157454378024 * (int(u'hey' in comment_text_words)),
           -0.02492608508166702791 * (int(u'hindus' in comment_text_words)),
           0.069601957465770664113 * (int(u'hip' in comment_text_words)),
           0.033841319158570359882 * (int(u'ho' in comment_text_words)),
          -0.034451541667740372132 * (int(u'holds' in comment_text_words)),
           -0.10653548634747861401 * (int(u'holidays' in comment_text_words)),
           0.071073790996848490442 * (int(u'hollow' in comment_text_words)),
          -0.074056594776953255099 * (int(u'homeland' in comment_text_words)),
          0.0088832737194134252234 * (int(u'hominem' in comment_text_words)),
          -0.050371554741483248741 * (int(u'homo' in comment_text_words)),
           -0.07669993860486011561 * (int(u'homosexuals' in comment_text_words)),
           0.056029706985521642026 * (int(u'honour' in comment_text_words)),
           -0.03464606312788818393 * (int(u'house' in comment_text_words)),
          -0.006419630367474444925 * (int(u'however' in comment_text_words)),
           0.011111465817369822057 * (int(u'http' in comment_text_words)),
           0.038093699256071970882 * (int(u'humour' in comment_text_words)),
         -0.0082952829246010616887 * (int(u'hungry' in comment_text_words)),
           0.017107724926276331179 * (int(u'hunt' in comment_text_words)),
          -0.044342272107009000903 * (int(u'hyperbole' in comment_text_words)),
           0.033784919986055261809 * (int(u'idiots' in comment_text_words)),
           0.018710904887474004432 * (int(u'ignoring' in comment_text_words)),
           -0.12782942204069486225 * (int(u'igor' in comment_text_words)),
           0.041730182052728430342 * (int(u'ill' in comment_text_words)),
           0.011763737803404124282 * (int(u'illegitimate' in comment_text_words)),
           0.034106287316448782865 * (int(u'illiterate' in comment_text_words)),
          -0.042597036206592872598 * (int(u'illness' in comment_text_words)),
           0.047582064029450574227 * (int(u'illusion' in comment_text_words)),
          -0.037851007642295916855 * (int(u'immature' in comment_text_words)),
           0.056870001938017222809 * (int(u'immediately' in comment_text_words)),
          -0.054021528766067479499 * (int(u'immigration' in comment_text_words)),
           -0.01266511990083380064 * (int(u'impartial' in comment_text_words)),
          -0.077309781337001740043 * (int(u'implied' in comment_text_words)),
          -0.030184185471501364523 * (int(u'importantly' in comment_text_words)),
          -0.052834274042294217255 * (int(u'impossible' in comment_text_words)),
          -0.032050693035226986149 * (int(u'industry' in comment_text_words)),
          -0.025545947113036062553 * (int(u'infinite' in comment_text_words)),
          -0.057474287451677055771 * (int(u'inflammatory' in comment_text_words)),
          -0.034643367697824695162 * (int(u'injustice' in comment_text_words)),
          -0.017419497571536830816 * (int(u'inside' in comment_text_words)),
           -0.19524525497867861734 * (int(u'insignificant' in comment_text_words)),
           0.011844063077020542882 * (int(u'insisting' in comment_text_words)),
           0.083478509151680077571 * (int(u'instantly' in comment_text_words)),
           0.062971308276863202646 * (int(u'intact' in comment_text_words)),
          0.0021577648539275991538 * (int(u'intelligent' in comment_text_words)),
          -0.051598117607632769388 * (int(u'intense' in comment_text_words)),
          -0.038457946706913867518 * (int(u'intentional' in comment_text_words)),
           -0.10468020649957766877 * (int(u'interact' in comment_text_words)),
           -0.04067519191902559117 * (int(u'interpreted' in comment_text_words)),
          -0.055010178488086702853 * (int(u'intimidate' in comment_text_words)),
          -0.016101818494282087862 * (int(u'iran' in comment_text_words)),
         -0.0031829301932499863732 * (int(u'ireland' in comment_text_words)),
          -0.017251983157845290906 * (int(u'irresponsible' in comment_text_words)),
         -0.0092060846544658615082 * (int(u'is' in comment_text_words)),
          -0.056322107957014269641 * (int(u'islands' in comment_text_words)),
           -0.13079344372440671052 * (int(u'isles' in comment_text_words)),
          -0.011870094897125104544 * (int(u'it' in comment_text_words)),
         -0.0047701461625369442551 * (int(u'jackass' in comment_text_words)),
           -0.02722402612584035389 * (int(u'jackson' in comment_text_words)),
          -0.075123608848179493358 * (int(u'jail' in comment_text_words)),
           0.086330787353359028646 * (int(u'jehochman' in comment_text_words)),
           0.018571115681456480673 * (int(u'jerks' in comment_text_words)),
           0.061643117233453573189 * (int(u'jew' in comment_text_words)),
           0.023685616416461108269 * (int(u'jewish' in comment_text_words)),
           0.023223785032160240011 * (int(u'jews' in comment_text_words)),
          -0.015541919015924785322 * (int(u'joe' in comment_text_words)),
          0.0007810904100881255847 * (int(u'joke' in comment_text_words)),
           0.058585276442999593971 * (int(u'jokes' in comment_text_words)),
          -0.011947848453723387482 * (int(u'joseph' in comment_text_words)),
           -0.10212153757636506513 * (int(u'judged' in comment_text_words)),
         -0.0084646866232821679765 * (int(u'jumping' in comment_text_words)),
           -0.35238584386104271351 * (int(u'jumps' in comment_text_words)),
           0.034259114935021935111 * (int(u'kate' in comment_text_words)),
          -0.021667155921048879968 * (int(u'key' in comment_text_words)),
           0.017614536942398343472 * (int(u'kicked' in comment_text_words)),
            0.01301410611591070042 * (int(u'kill' in comment_text_words)),
         -0.0057563530228381910511 * (int(u'killing' in comment_text_words)),
          -0.013304667303023483965 * (int(u'kindly' in comment_text_words)),
          -0.016323151903529075496 * (int(u'kiss' in comment_text_words)),
          -0.046547409598991090762 * (int(u'kite' in comment_text_words)),
          -0.067265249854112710293 * (int(u'klan' in comment_text_words)),
          -0.069781159858428629117 * (int(u'knee' in comment_text_words)),
          -0.021818375423547330116 * (int(u'known' in comment_text_words)),
          -0.024135333874212221844 * (int(u'later' in comment_text_words)),
           0.086644734121087293999 * (int(u'laughing' in comment_text_words)),
           0.007545861913532158835 * (int(u'lazy' in comment_text_words)),
           -0.02915141200707916172 * (int(u'le' in comment_text_words)),
          -0.056891484838787788336 * (int(u'league' in comment_text_words)),
          -0.022794223425964731111 * (int(u'leap' in comment_text_words)),
           0.085619900710507068631 * (int(u'legion' in comment_text_words)),
           0.014243304048121697633 * (int(u'lesbian' in comment_text_words)),
           0.048780458661310178992 * (int(u'lgbt' in comment_text_words)),
          -0.084385889177444065035 * (int(u'liars' in comment_text_words)),
          -0.027493754496979123242 * (int(u'life' in comment_text_words)),
            0.14778472766151751205 * (int(u'lighting' in comment_text_words)),
          -0.024879212962622705013 * (int(u'like' in comment_text_words)),
           0.054428045387242079967 * (int(u'linda' in comment_text_words)),
           0.027757870887916286096 * (int(u'listas' in comment_text_words)),
          -0.028296155783550453605 * (int(u'listen' in comment_text_words)),
          -0.039478011124942757881 * (int(u'little' in comment_text_words)),
           0.049548611425148332554 * (int(u'lmao' in comment_text_words)),
           0.039463154650777550192 * (int(u'lodge' in comment_text_words)),
          -0.012048201979385898472 * (int(u'looked' in comment_text_words)),
           -0.12658513042957728056 * (int(u'loose' in comment_text_words)),
           -0.01008190235665620306 * (int(u'loser' in comment_text_words)),
          0.0020425059863292752138 * (int(u'losers' in comment_text_words)),
           0.032734592130975771751 * (int(u'loss' in comment_text_words)),
          -0.029643646821309835304 * (int(u'lovely' in comment_text_words)),
          -0.070932000162531747001 * (int(u'loving' in comment_text_words)),
          -0.042692095288165927969 * (int(u'luckily' in comment_text_words)),
           0.017241993642163656181 * (int(u'lying' in comment_text_words)),
          -0.057377584145682399008 * (int(u'macedonians' in comment_text_words)),
          -0.033491419535484512338 * (int(u'mad' in comment_text_words)),
          -0.018136261023906963957 * (int(u'male' in comment_text_words)),
          -0.029725477616635593431 * (int(u'males' in comment_text_words)),
           0.013599880186747620794 * (int(u'malicious' in comment_text_words)),
           0.044957648774916876555 * (int(u'mama' in comment_text_words)),
           0.071488666135413617142 * (int(u'manchester' in comment_text_words)),
         0.00040286991761132067607 * (int(u'manga' in comment_text_words)),
          -0.042633812962559226867 * (int(u'mankind' in comment_text_words)),
          -0.084670922227783010361 * (int(u'mason' in comment_text_words)),
          0.0018029626045297565164 * (int(u'massive' in comment_text_words)),
          -0.037580151576809654734 * (int(u'masters' in comment_text_words)),
           0.023113254019684736223 * (int(u'mate' in comment_text_words)),
           0.055462686399496252676 * (int(u'math' in comment_text_words)),
            0.01351325315127671102 * (int(u'mathematics' in comment_text_words)),
           -0.10720410520221197725 * (int(u'matthew' in comment_text_words)),
          0.0013807808479844748111 * (int(u'meat' in comment_text_words)),
          0.0021844217971927325922 * (int(u'medals' in comment_text_words)),
          -0.042586298812910004796 * (int(u'media' in comment_text_words)),
          -0.044831899079983028589 * (int(u'mega' in comment_text_words)),
           0.074981540623348857943 * (int(u'mental' in comment_text_words)),
            0.10518078872225411491 * (int(u'mentality' in comment_text_words)),
          -0.042535695956129028483 * (int(u'method' in comment_text_words)),
            0.26751590914120276787 * (int(u'metropolitan' in comment_text_words)),
           0.027512219392185167238 * (int(u'mighty' in comment_text_words)),
           0.093009666876606902908 * (int(u'minded' in comment_text_words)),
           0.033101431356150172458 * (int(u'miserable' in comment_text_words)),
          -0.031805029082583508027 * (int(u'misery' in comment_text_words)),
          -0.026810585199697371112 * (int(u'missing' in comment_text_words)),
           0.054096291506778935998 * (int(u'mommy' in comment_text_words)),
          0.0046321938444628925693 * (int(u'more' in comment_text_words)),
           -0.11235801015818944193 * (int(u'morgan' in comment_text_words)),
           0.075387537593888567788 * (int(u'moronic' in comment_text_words)),
           0.051147714179539373325 * (int(u'morons' in comment_text_words)),
          -0.043750364745672329359 * (int(u'mostly' in comment_text_words)),
          -0.022054411279731581108 * (int(u'motivated' in comment_text_words)),
          0.0065137172084550271425 * (int(u'motive' in comment_text_words)),
         -0.0017913419211437074316 * (int(u'mouth' in comment_text_words)),
          -0.037472512713528374761 * (int(u'mtv' in comment_text_words)),
           -0.03042291704657374557 * (int(u'multiple' in comment_text_words)),
          -0.064321246879849566791 * (int(u'murderer' in comment_text_words)),
          -0.019415639506514415641 * (int(u'na' in comment_text_words)),
           0.018254666757251125642 * (int(u'names' in comment_text_words)),
           0.077488856933561375828 * (int(u'nato' in comment_text_words)),
          -0.026204292102132951958 * (int(u'nazi' in comment_text_words)),
           -0.15627413263776485097 * (int(u'nd' in comment_text_words)),
         -0.0075517982060130089267 * (int(u'nearly' in comment_text_words)),
           0.073915458328235414398 * (int(u'neat' in comment_text_words)),
          -0.012267400898920532981 * (int(u'neck' in comment_text_words)),
          -0.012854547375155978353 * (int(u'negro' in comment_text_words)),
           0.025523824247920941233 * (int(u'nerds' in comment_text_words)),
          -0.071878710840871495868 * (int(u'nerve' in comment_text_words)),
          -0.087741495192626139943 * (int(u'networking' in comment_text_words)),
           0.059687511255596001347 * (int(u'newcomer' in comment_text_words)),
          -0.059304952699012813966 * (int(u'nicholas' in comment_text_words)),
          -0.034406312510979877861 * (int(u'nigga' in comment_text_words)),
           0.036372908989169949745 * (int(u'nonsense' in comment_text_words)),
          -0.051429158618261909541 * (int(u'noone' in comment_text_words)),
            0.12584933969778244744 * (int(u'nu' in comment_text_words)),
           0.080678273610067241517 * (int(u'numerical' in comment_text_words)),
           0.057759965012329876621 * (int(u'nut' in comment_text_words)),
            0.03347964059413867377 * (int(u'obama' in comment_text_words)),
          -0.053901054563661142394 * (int(u'obnoxious' in comment_text_words)),
          -0.064751129493521464298 * (int(u'obscene' in comment_text_words)),
           -0.14034174366057142191 * (int(u'obsolete' in comment_text_words)),
          -0.069336752493830366983 * (int(u'occured' in comment_text_words)),
          -0.046554353766708765627 * (int(u'october' in comment_text_words)),
           -0.02592203128846074997 * (int(u'offending' in comment_text_words)),
          -0.020225765846307593165 * (int(u'officer' in comment_text_words)),
          -0.046286147374599205528 * (int(u'ohio' in comment_text_words)),
          0.0019809992556606911829 * (int(u'oi' in comment_text_words)),
          -0.077323070007975003293 * (int(u'olympics' in comment_text_words)),
         -0.0099898202502532711805 * (int(u'opinion' in comment_text_words)),
          -0.033577794322557065476 * (int(u'opinions' in comment_text_words)),
           0.050647696993446233993 * (int(u'oppression' in comment_text_words)),
           0.022843268096262817068 * (int(u'oral' in comment_text_words)),
           0.054001286652300500846 * (int(u'orientation' in comment_text_words)),
           -0.02676874800862167128 * (int(u'outside' in comment_text_words)),
          -0.089343023767358004106 * (int(u'overboard' in comment_text_words)),
          -0.011794447396117149068 * (int(u'overwhelming' in comment_text_words)),
            0.12663917652292275284 * (int(u'overzealous' in comment_text_words)),
         -0.0047302647060413023292 * (int(u'own' in comment_text_words)),
         -0.0079726623637691867996 * (int(u'pain' in comment_text_words)),
           -0.04137994956567005167 * (int(u'painting' in comment_text_words)),
          -0.009891443749131499652 * (int(u'paranoid' in comment_text_words)),
          -0.026654375471450362373 * (int(u'parents' in comment_text_words)),
          -0.018891237451212367776 * (int(u'particular' in comment_text_words)),
           0.067239001242035412176 * (int(u'pedophile' in comment_text_words)),
           0.071211981212972216837 * (int(u'peers' in comment_text_words)),
         -0.0079522177779122668606 * (int(u'people' in comment_text_words)),
          0.0037068559973909417579 * (int(u'peoples' in comment_text_words)),
         -0.0064141022532962308628 * (int(u'perform' in comment_text_words)),
           0.098638269183280308239 * (int(u'persistent' in comment_text_words)),
          0.0076178088700934384891 * (int(u'person' in comment_text_words)),
          -0.053613120417192718092 * (int(u'personality' in comment_text_words)),
         -0.0053130484079827681315 * (int(u'ph' in comment_text_words)),
           0.056438211754291416067 * (int(u'phase' in comment_text_words)),
           0.070142271684968490741 * (int(u'photography' in comment_text_words)),
          -0.019847891960666129429 * (int(u'picking' in comment_text_words)),
           0.075252558449784248684 * (int(u'pipe' in comment_text_words)),
          -0.038085573402206737703 * (int(u'pit' in comment_text_words)),
           0.020473791198743668956 * (int(u'pity' in comment_text_words)),
           -0.07748953964859107979 * (int(u'pl' in comment_text_words)),
           0.020888947610297269042 * (int(u'planet' in comment_text_words)),
           0.034084618893354887148 * (int(u'plot' in comment_text_words)),
            0.10441567399421895768 * (int(u'plots' in comment_text_words)),
            0.03349727054669260351 * (int(u'politics' in comment_text_words)),
           -0.01956662794311394421 * (int(u'pompous' in comment_text_words)),
           0.063692578239934710682 * (int(u'poo' in comment_text_words)),
          0.0081926493308767476403 * (int(u'poop' in comment_text_words)),
           0.010452630813443520089 * (int(u'pop' in comment_text_words)),
          -0.047522833996056577932 * (int(u'pornography' in comment_text_words)),
           0.049233963695784414838 * (int(u'posts' in comment_text_words)),
          -0.091555683234657236902 * (int(u'potter' in comment_text_words)),
           0.009878253064908802325 * (int(u'power' in comment_text_words)),
           0.045987574041849214879 * (int(u'ppl' in comment_text_words)),
           -0.01646964270917631451 * (int(u'practice' in comment_text_words)),
         -0.0071298658681612026369 * (int(u'practices' in comment_text_words)),
            0.17540384558319532804 * (int(u'practicing' in comment_text_words)),
          -0.004194583668491884236 * (int(u'pray' in comment_text_words)),
          -0.056107212706968799532 * (int(u'prayer' in comment_text_words)),
           0.029686460349861071989 * (int(u'preaching' in comment_text_words)),
            0.12926266779735109957 * (int(u'predominantly' in comment_text_words)),
           0.016866406960688997557 * (int(u'premise' in comment_text_words)),
          -0.033918317920911482399 * (int(u'prestigious' in comment_text_words)),
          -0.030809049097474013046 * (int(u'prevail' in comment_text_words)),
          -0.023091547531788710751 * (int(u'priests' in comment_text_words)),
           0.014001715502606508143 * (int(u'prob' in comment_text_words)),
          -0.062002762830489777102 * (int(u'proclaimed' in comment_text_words)),
         -0.0092906280532598414096 * (int(u'produced' in comment_text_words)),
          -0.086751205818385385138 * (int(u'profile' in comment_text_words)),
          -0.091741669728122809957 * (int(u'promoting' in comment_text_words)),
          -0.075093744511547280696 * (int(u'promptly' in comment_text_words)),
          -0.049894949875262592509 * (int(u'prone' in comment_text_words)),
           0.036492738868996336954 * (int(u'proof' in comment_text_words)),
           0.038913228375827446648 * (int(u'proven' in comment_text_words)),
           0.047542554980670746756 * (int(u'proving' in comment_text_words)),
          0.0066389524809702168268 * (int(u'ps' in comment_text_words)),
            0.04675393533694316428 * (int(u'pseudo' in comment_text_words)),
           0.041584422039977794738 * (int(u'psychological' in comment_text_words)),
          -0.015010373713517643005 * (int(u'public' in comment_text_words)),
           0.056982704682436907673 * (int(u'publicly' in comment_text_words)),
            0.01641678000972749546 * (int(u'puff' in comment_text_words)),
            0.08112479951403596401 * (int(u'punch' in comment_text_words)),
           0.013776665989851387059 * (int(u'punish' in comment_text_words)),
         -0.0091761335515229934534 * (int(u'punishment' in comment_text_words)),
           0.012187546484399524455 * (int(u'puppet' in comment_text_words)),
          -0.029627147981367597807 * (int(u'purchase' in comment_text_words)),
         -0.0054360191540920735701 * (int(u'pure' in comment_text_words)),
             0.0583456038639690866 * (int(u'purple' in comment_text_words)),
          -0.078973259752074170814 * (int(u'purposely' in comment_text_words)),
         -0.0013927093193388513735 * (int(u'qualification' in comment_text_words)),
          -0.061405483062812404871 * (int(u'quiet' in comment_text_words)),
          -0.051184424886897318674 * (int(u'quietly' in comment_text_words)),
          -0.029153713666030742213 * (int(u'quit' in comment_text_words)),
          -0.023129144833835237755 * (int(u'rants' in comment_text_words)),
           0.067264633512563232198 * (int(u'rat' in comment_text_words)),
         -0.0011793850684302883709 * (int(u're' in comment_text_words)),
          -0.022512944771773752556 * (int(u'reacting' in comment_text_words)),
          -0.016086209657630877456 * (int(u'reading' in comment_text_words)),
          -0.015588835186459084822 * (int(u'really' in comment_text_words)),
            0.13854368410760187214 * (int(u'reckon' in comment_text_words)),
          -0.025902695364526440169 * (int(u'redirected' in comment_text_words)),
          -0.080415428316401346698 * (int(u'refered' in comment_text_words)),
          -0.029395777563500584406 * (int(u'reform' in comment_text_words)),
          0.0065203673017551703334 * (int(u'refuses' in comment_text_words)),
           0.023144578583433411678 * (int(u'refusing' in comment_text_words)),
         -0.0069663207700380452847 * (int(u'regarding' in comment_text_words)),
           -0.06220072577500339317 * (int(u'regular' in comment_text_words)),
          -0.056770260461716008649 * (int(u'relate' in comment_text_words)),
           -0.03033955558053584925 * (int(u'reliably' in comment_text_words)),
          -0.036164512196068566985 * (int(u'remind' in comment_text_words)),
         -0.0091322813765268000957 * (int(u'reminded' in comment_text_words)),
           0.017428072692966853352 * (int(u'reminds' in comment_text_words)),
           0.041824248835802699253 * (int(u'removal' in comment_text_words)),
           0.026078044250672550303 * (int(u'removing' in comment_text_words)),
            0.30210608962701884783 * (int(u'rep' in comment_text_words)),
           0.025980910925281631985 * (int(u'repeatedly' in comment_text_words)),
           0.012145423724265523233 * (int(u'replies' in comment_text_words)),
          -0.047614598483817008323 * (int(u'reporting' in comment_text_words)),
          -0.032158780822588189929 * (int(u'reputation' in comment_text_words)),
          0.0099155379172937777033 * (int(u'research' in comment_text_words)),
          -0.085780093105915930507 * (int(u'resort' in comment_text_words)),
          -0.021448762385703149663 * (int(u'resorting' in comment_text_words)),
          -0.012312838454662848597 * (int(u'respond' in comment_text_words)),
           0.031196602140521539209 * (int(u'responding' in comment_text_words)),
         -0.0065134820363126526635 * (int(u'responses' in comment_text_words)),
            0.02792430596658550257 * (int(u'retards' in comment_text_words)),
           -0.11269000688634379925 * (int(u'retiring' in comment_text_words)),
           0.060229912051603821943 * (int(u'revenge' in comment_text_words)),
          0.0072018040126744032117 * (int(u'reverting' in comment_text_words)),
           0.080918501028580169798 * (int(u'reword' in comment_text_words)),
            0.11683082003211699151 * (int(u'rice' in comment_text_words)),
         -0.0023592032638543505157 * (int(u'rid' in comment_text_words)),
          -0.039451493827930714242 * (int(u'ride' in comment_text_words)),
         -0.0072577529239539655473 * (int(u'right' in comment_text_words)),
         -0.0041800951176575442889 * (int(u'rings' in comment_text_words)),
            0.06878631598436799055 * (int(u'rob' in comment_text_words)),
          -0.050090603751730952697 * (int(u'rocket' in comment_text_words)),
           0.017883384994102382443 * (int(u'rocks' in comment_text_words)),
           0.046963251850324835845 * (int(u'rogue' in comment_text_words)),
            0.15602255952605276201 * (int(u'rubber' in comment_text_words)),
           0.009816103171581901099 * (int(u'rudeness' in comment_text_words)),
           0.045184185632835056901 * (int(u'ruining' in comment_text_words)),
            0.10554009617625040307 * (int(u'rulers' in comment_text_words)),
          0.0091008188262495046811 * (int(u'rules' in comment_text_words)),
         -0.0085314878345510644642 * (int(u'ruling' in comment_text_words)),
          -0.027268686095574094091 * (int(u'russians' in comment_text_words)),
           0.013175232961896158473 * (int(u'ryulong' in comment_text_words)),
          -0.062512896390182981499 * (int(u'sacred' in comment_text_words)),
         -0.0046265154689756476936 * (int(u'sad' in comment_text_words)),
          -0.045871171406152558281 * (int(u'sadly' in comment_text_words)),
          -0.036173093308879522567 * (int(u'sake' in comment_text_words)),
           0.083568508931848597965 * (int(u'sale' in comment_text_words)),
          -0.010470189834677991192 * (int(u'salvador' in comment_text_words)),
           0.053994382726155386309 * (int(u'sandy' in comment_text_words)),
           -0.06582568302232773283 * (int(u'sarah' in comment_text_words)),
           0.018964681854452280357 * (int(u'sat' in comment_text_words)),
          -0.036303027026101414299 * (int(u'savage' in comment_text_words)),
          -0.016169638749060150779 * (int(u'say' in comment_text_words)),
           0.036237698365985532289 * (int(u'scandal' in comment_text_words)),
         -0.0051443750023960205242 * (int(u'scared' in comment_text_words)),
          -0.088879732293704202806 * (int(u'scenes' in comment_text_words)),
          -0.024550360024810456011 * (int(u'schedule' in comment_text_words)),
          -0.084177010131307919427 * (int(u'screaming' in comment_text_words)),
           0.059779804914637364732 * (int(u'scroll' in comment_text_words)),
           0.083147151439123614947 * (int(u'scumbag' in comment_text_words)),
           0.077184468277494852617 * (int(u'sec' in comment_text_words)),
           -0.01434737874430905627 * (int(u'second' in comment_text_words)),
          -0.038658298278794786529 * (int(u'secrets' in comment_text_words)),
          0.0099718276076562646637 * (int(u'self' in comment_text_words)),
          -0.065412552155402051146 * (int(u'semen' in comment_text_words)),
          0.0087792623070335647217 * (int(u'semi' in comment_text_words)),
             0.0250401544208878786 * (int(u'semitic' in comment_text_words)),
         -0.0042346683828697129562 * (int(u'send' in comment_text_words)),
          -0.050802069986827957937 * (int(u'sending' in comment_text_words)),
         -0.0013431674190633670525 * (int(u'separately' in comment_text_words)),
           -0.03510191342665626435 * (int(u'serbia' in comment_text_words)),
          -0.040400183061564350073 * (int(u'sh' in comment_text_words)),
           0.093172212867498938049 * (int(u'shadow' in comment_text_words)),
          -0.024347663081216389808 * (int(u'shall' in comment_text_words)),
           0.061011845360465187527 * (int(u'shallow' in comment_text_words)),
           0.015165495104400643703 * (int(u'shame' in comment_text_words)),
          -0.010216116017326045892 * (int(u'shameful' in comment_text_words)),
          0.0027603025198714220209 * (int(u'shameless' in comment_text_words)),
           0.019475159953481487041 * (int(u'shell' in comment_text_words)),
          -0.078455022940546390031 * (int(u'shopping' in comment_text_words)),
          -0.070018365751425643007 * (int(u'shoulder' in comment_text_words)),
           -0.15173278777034807097 * (int(u'shower' in comment_text_words)),
           0.020444008533605013966 * (int(u'simply' in comment_text_words)),
          -0.023906170519367372918 * (int(u'sin' in comment_text_words)),
           0.065108225423534532572 * (int(u'sink' in comment_text_words)),
          -0.067604889908950485666 * (int(u'sit' in comment_text_words)),
           0.093966936389620545067 * (int(u'slapped' in comment_text_words)),
          -0.011079700888872714634 * (int(u'slave' in comment_text_words)),
            0.08410924922452631125 * (int(u'slaves' in comment_text_words)),
         -0.0064962697761993043119 * (int(u'sloppy' in comment_text_words)),
           0.031976609853210227752 * (int(u'slurs' in comment_text_words)),
          0.0028527041139887417999 * (int(u'smart' in comment_text_words)),
          -0.065007938244811630524 * (int(u'smash' in comment_text_words)),
           0.051601857278298503273 * (int(u'soap' in comment_text_words)),
          -0.046353757971952395944 * (int(u'soapboxing' in comment_text_words)),
           0.044783128530864656225 * (int(u'sockpuppet' in comment_text_words)),
          0.0031858669536048429653 * (int(u'soldiers' in comment_text_words)),
           0.021103217022295370914 * (int(u'sole' in comment_text_words)),
           0.099709006752739082202 * (int(u'someones' in comment_text_words)),
           0.052173344630198445482 * (int(u'sometime' in comment_text_words)),
          -0.018897317184660068767 * (int(u'son' in comment_text_words)),
          -0.015022799238830068444 * (int(u'sooner' in comment_text_words)),
          -0.011449324265580733059 * (int(u'sorry' in comment_text_words)),
           0.012346690317235441484 * (int(u'soul' in comment_text_words)),
        -0.00044445192543787314775 * (int(u'sovereignty' in comment_text_words)),
           -0.07141785833428077046 * (int(u'soviet' in comment_text_words)),
          -0.011766968963972957926 * (int(u'spam' in comment_text_words)),
          -0.030006290216401668852 * (int(u'speaks' in comment_text_words)),
            0.23500502151480218149 * (int(u'spider' in comment_text_words)),
          -0.024534868809018957336 * (int(u'spurious' in comment_text_words)),
          -0.061525533850083136134 * (int(u'squat' in comment_text_words)),
            0.10560721403514375383 * (int(u'stadium' in comment_text_words)),
          -0.051155400789589369759 * (int(u'stark' in comment_text_words)),
           -0.10161387296707424799 * (int(u'stars' in comment_text_words)),
        -0.00034401039468658111461 * (int(u'started' in comment_text_words)),
         -0.0098487989651366292243 * (int(u'stay' in comment_text_words)),
          -0.088227466533322293496 * (int(u'stealing' in comment_text_words)),
          -0.012380892721725621219 * (int(u'still' in comment_text_words)),
          -0.034283980524756882402 * (int(u'stole' in comment_text_words)),
          -0.012059507922053309265 * (int(u'stop' in comment_text_words)),
           0.026602878791160678579 * (int(u'stops' in comment_text_words)),
           -0.11808177809399800218 * (int(u'stranger' in comment_text_words)),
          -0.046548396525135184165 * (int(u'straw' in comment_text_words)),
          -0.079385795295924058146 * (int(u'stuffed' in comment_text_words)),
          -0.037930166827704142607 * (int(u'stupid' in comment_text_words)),
         -0.0065828778197730315747 * (int(u'such' in comment_text_words)),
         -0.0061690590960430070552 * (int(u'sue' in comment_text_words)),
           -0.17020645687348634478 * (int(u'sugar' in comment_text_words)),
          -0.019485727359113365342 * (int(u'suit' in comment_text_words)),
            0.13416777557521197828 * (int(u'sunni' in comment_text_words)),
           0.076990050332548007339 * (int(u'sunrise' in comment_text_words)),
           0.041621395855642139006 * (int(u'supporters' in comment_text_words)),
          -0.027837035507494969216 * (int(u'surface' in comment_text_words)),
           0.017378410738719304862 * (int(u'surprise' in comment_text_words)),
            0.10043164784381572452 * (int(u'suspicion' in comment_text_words)),
           0.091784860498098572168 * (int(u'swearing' in comment_text_words)),
          -0.019975639989414283038 * (int(u'sweat' in comment_text_words)),
           0.053014388425888646361 * (int(u'sympathetic' in comment_text_words)),
            0.10313761846668006994 * (int(u'sympathy' in comment_text_words)),
         -0.0035473129100708445868 * (int(u'system' in comment_text_words)),
          -0.093431819219222472794 * (int(u'tail' in comment_text_words)),
          -0.010004004995941324571 * (int(u'take' in comment_text_words)),
            -0.1071928646605671831 * (int(u'talented' in comment_text_words)),
          0.0085880590944604970338 * (int(u'talk' in comment_text_words)),
          0.0063556445784419633022 * (int(u'talking' in comment_text_words)),
           -0.10806505323620298198 * (int(u'tastes' in comment_text_words)),
            0.14069666142939507569 * (int(u'tedious' in comment_text_words)),
          0.0095346243780221155706 * (int(u'terrible' in comment_text_words)),
          0.0083032211476136968259 * (int(u'terrorists' in comment_text_words)),
          -0.018824088082114252135 * (int(u'thanx' in comment_text_words)),
           0.002187706089266086909 * (int(u'that' in comment_text_words)),
           0.043246983672180182301 * (int(u'thee' in comment_text_words)),
          -0.050554049192315167471 * (int(u'therapy' in comment_text_words)),
          -0.036545853648311740769 * (int(u'theres' in comment_text_words)),
            0.09489009376688162245 * (int(u'thief' in comment_text_words)),
          -0.013737438912586287174 * (int(u'thinking' in comment_text_words)),
          0.0058048795757677743717 * (int(u'this' in comment_text_words)),
          -0.016468149270662844202 * (int(u'though' in comment_text_words)),
          -0.023279213739672267869 * (int(u'threatened' in comment_text_words)),
          -0.015546304777277955914 * (int(u'three' in comment_text_words)),
           0.011407439468175727473 * (int(u'tight' in comment_text_words)),
          -0.075183099431874789165 * (int(u'til' in comment_text_words)),
        -0.00089184148186215476614 * (int(u'times' in comment_text_words)),
           0.018805705307799673587 * (int(u'tiny' in comment_text_words)),
          0.0015707373299873451759 * (int(u'to' in comment_text_words)),
          -0.061697458495117991906 * (int(u'tokyo' in comment_text_words)),
          -0.017515071733443958457 * (int(u'tolerate' in comment_text_words)),
            0.11941844880190485845 * (int(u'tolerated' in comment_text_words)),
           -0.03965173954710857207 * (int(u'tom' in comment_text_words)),
           0.015999530662430188482 * (int(u'tomorrow' in comment_text_words)),
          -0.077683854563230111956 * (int(u'topical' in comment_text_words)),
          -0.031297531903533015729 * (int(u'tough' in comment_text_words)),
          -0.011714593505097429735 * (int(u'trade' in comment_text_words)),
          -0.089122353799647613393 * (int(u'tragic' in comment_text_words)),
         -0.0022439523419595869616 * (int(u'trash' in comment_text_words)),
           0.016667885410434488819 * (int(u'troll' in comment_text_words)),
          -0.015437581157124636785 * (int(u'trolling' in comment_text_words)),
         -0.0077781577599968414541 * (int(u'trolls' in comment_text_words)),
          -0.011778069370900591875 * (int(u'troubling' in comment_text_words)),
           0.064421546546177155257 * (int(u'truck' in comment_text_words)),
           -0.02017370054732697468 * (int(u'true' in comment_text_words)),
          0.0084646023969657726432 * (int(u'truly' in comment_text_words)),
         -0.0094610988427929858247 * (int(u'truth' in comment_text_words)),
           -0.12900500851279134151 * (int(u'tu' in comment_text_words)),
           0.074500415756479046459 * (int(u'turd' in comment_text_words)),
          -0.085516641718893865454 * (int(u'turk' in comment_text_words)),
          0.0064698508566018650412 * (int(u'turning' in comment_text_words)),
          -0.024333720145330653417 * (int(u'turns' in comment_text_words)),
           -0.02330361330631268782 * (int(u'tw' in comment_text_words)),
           0.082255917390066268791 * (int(u'twisted' in comment_text_words)),
          -0.035787008167611100706 * (int(u'twitter' in comment_text_words)),
          -0.035630482851612389761 * (int(u'typical' in comment_text_words)),
          -0.020931344834663297844 * (int(u'ugly' in comment_text_words)),
          -0.028186893601296551803 * (int(u'uh' in comment_text_words)),
          -0.010970977330929715735 * (int(u'ultimately' in comment_text_words)),
          -0.021956892834211171051 * (int(u'unblocked' in comment_text_words)),
           0.011734492835908950578 * (int(u'uncomfortable' in comment_text_words)),
          -0.064731890262361493904 * (int(u'underground' in comment_text_words)),
          -0.089081363339003488688 * (int(u'understands' in comment_text_words)),
           0.013982544407350446306 * (int(u'unregistered' in comment_text_words)),
          -0.098787808654201642033 * (int(u'untill' in comment_text_words)),
          -0.050340757190587649694 * (int(u'unworthy' in comment_text_words)),
        -0.00087691829637872784325 * (int(u'ur' in comment_text_words)),
          -0.045115751716023166917 * (int(u'urgent' in comment_text_words)),
          -0.019839127952037103469 * (int(u'useless' in comment_text_words)),
          -0.032966469956059556157 * (int(u'username' in comment_text_words)),
           0.019847429371323103975 * (int(u'userpage' in comment_text_words)),
           0.010564128207288690439 * (int(u'utc' in comment_text_words)),
           -0.13821088946240925321 * (int(u'vagina' in comment_text_words)),
            0.21010082111564856722 * (int(u'vain' in comment_text_words)),
           -0.01400883849442840709 * (int(u'vendetta' in comment_text_words)),
         -0.0082846218753605867929 * (int(u'versa' in comment_text_words)),
          -0.047718570360501630367 * (int(u'verse' in comment_text_words)),
          -0.019081573463267473972 * (int(u'vested' in comment_text_words)),
          -0.012429039072592695109 * (int(u'victims' in comment_text_words)),
          -0.017890929524308669257 * (int(u'videos' in comment_text_words)),
          -0.013152720606732172873 * (int(u'views' in comment_text_words)),
          -0.093824361447890983201 * (int(u'vile' in comment_text_words)),
          -0.019138319284801003084 * (int(u'violations' in comment_text_words)),
          -0.035258684771569029359 * (int(u'wa' in comment_text_words)),
          0.0022572779249170385994 * (int(u'wales' in comment_text_words)),
           0.058344080415227053682 * (int(u'walls' in comment_text_words)),
           -0.10578574191530713822 * (int(u'wanker' in comment_text_words)),
         -0.0056543606060372618091 * (int(u'want' in comment_text_words)),
            0.02213363803667920543 * (int(u'warm' in comment_text_words)),
          -0.084086118129985065739 * (int(u'warming' in comment_text_words)),
           0.017054349093584681568 * (int(u'warning' in comment_text_words)),
           0.043052417961838992944 * (int(u'warrior' in comment_text_words)),
           0.031041164236161555151 * (int(u'washed' in comment_text_words)),
         0.00081155429650952681363 * (int(u'waves' in comment_text_words)),
        -0.00056845528297912346904 * (int(u'wearing' in comment_text_words)),
           0.070369658461883977107 * (int(u'welcomed' in comment_text_words)),
          -0.031957599031395812761 * (int(u'wherever' in comment_text_words)),
         -0.0094443013397280464738 * (int(u'who' in comment_text_words)),
            0.12707534748925863877 * (int(u'whoops' in comment_text_words)),
          0.0069204793060175344432 * (int(u'wikipedia' in comment_text_words)),
          -0.076244953488480482484 * (int(u'wikipeida' in comment_text_words)),
         -0.0089751881778500665343 * (int(u'wikis' in comment_text_words)),
          0.0033358879690206085997 * (int(u'will' in comment_text_words)),
          -0.039286188031874232085 * (int(u'winner' in comment_text_words)),
          -0.035220508326393104581 * (int(u'winners' in comment_text_words)),
         -0.0096382308812835576495 * (int(u'wisdom' in comment_text_words)),
         -0.0066511029848931816494 * (int(u'wish' in comment_text_words)),
          -0.024801326404582026142 * (int(u'wit' in comment_text_words)),
           0.016769708200830069078 * (int(u'wonder' in comment_text_words)),
          -0.083494811841577226685 * (int(u'wont' in comment_text_words)),
          -0.017031159891761658148 * (int(u'works' in comment_text_words)),
         -0.0091848690497433518182 * (int(u'world' in comment_text_words)),
           0.036438032976427871257 * (int(u'wouldnt' in comment_text_words)),
          -0.025413911857540345124 * (int(u'wow' in comment_text_words)),
         -0.0043893940320546814132 * (int(u'wreck' in comment_text_words)),
         -0.0026983392720735799308 * (int(u'write' in comment_text_words)),
          -0.047931000343628314686 * (int(u'writings' in comment_text_words)),
           0.059036720064188710766 * (int(u'xd' in comment_text_words)),
          -0.024323509587620119332 * (int(u'year' in comment_text_words)),
         -0.0017499303699007513237 * (int(u'yesterday' in comment_text_words)),
         -0.0053061639771656054013 * (int(u'you' in comment_text_words)),
           0.025232680702578398624 * (int(u'young' in comment_text_words)),
          -0.017223646169259396871 * (int(u'zionist' in comment_text_words)),
          -0.042998269939617916879 * (round_tfidf_word_stop_lr_toxic),
           -0.40916906919432954881 * (round_tfidf_word_stop_lr_threat),
             1.1175892822395823156 * (round_tfidf_word_stop_lr_insult),
           -0.12244119009812742815 * (round_tfidf_word_stop_lr_identity_hate),
            0.31934297468797012698 * (round_tfidf_word_stop_nblr_toxic),
            0.14325115261822515822 * (round_tfidf_word_stop_nblr_obscene),
          -0.019451498563425251992 * (round_tfidf_word_stop_nblr_insult),
           0.069119207251948783233 * (round_tfidf_word_stop_nblr_identity_hate),
           -0.15276271820510098354 * (round_tfidf_word_nostop_lr_insult),
           -0.14683057208063809984 * (round_tfidf_word_nostop_lr_identity_hate),
             0.1447549198297093942 * (round_tfidf_word_nostop_nblr_toxic),
            0.33741498687731402706 * (round_tfidf_word_nostop_nblr_obscene),
            0.10305319135348124659 * (round_tfidf_word_nostop_nblr_identity_hate),
             0.3802966115475771014 * (round_tfidf_char_lr_toxic),
            0.37404346477668010129 * (round_tfidf_char_lr_obscene),
          -0.012474624211558480932 * (round_tfidf_char_lr_insult),
           -0.37513028915074253522 * (round_tfidf_char_lr_identity_hate),
            0.29333938990537061775 * (round_tfidf_char_nblr_toxic),
            0.11604541436923550279 * (round_tfidf_char_nblr_obscene),
            0.41646337022299534381 * (round_tfidf_union_lr_toxic),
            0.55293691726095361982 * (round_tfidf_union_lr_obscene),
           0.083401546367996154396 * (round_tfidf_union_lr_insult),
            0.39105291173708184305 * (round_tfidf_union_nblr_toxic),
            0.17165386655927772352 * (round_tfidf_union_nblr_obscene),
            0.31290624464210042843 * (round_tfidf_union_nblr_threat),
           0.035270209593101814471 * (round_tfidf_union_nblr_insult),
         4.9674491599245454779E-05 * (round_num_lowercase),
            0.46955975530956406416 * (round_capital_per_char),
           -0.02275594741465123208 * (round_lowercase_per_char),
           -0.34461012850554489928 * (round_punctuation_per_char),
         0.00010917628035355106994 * (round_num_words_upper),
         0.00051185024614254894318 * (round_num_words_title),
           0.043233974965245024202 * (round_tfidf_word_stop_nblr_toxic <= 0.27129608392715454 and 
                                     round_tfidf_union_lr_toxic <= 0.51140213012695312 and 
                                     round_tfidf_union_nblr_toxic > 0.13555708527565002),
           -0.12003711993949862935 * (round_tfidf_char_lr_toxic <= 0.020266205072402954 and 
                                     round_tfidf_union_lr_toxic <= 0.027191130444407463 and 
                                     round_tfidf_union_nblr_toxic <= 0.13555708527565002),
           0.048078666134641984131 * (round_tfidf_word_stop_nblr_toxic > 0.91490352153778076 and 
                                     round_tfidf_union_lr_toxic > 0.50629949569702148 and 
                                     round_tfidf_union_nblr_toxic > 0.11764827370643616),
           0.029550196841423577443 * (round_tfidf_char_nblr_toxic > 0.58039271831512451 and 
                                     round_tfidf_union_lr_toxic > 0.85663366317749023),
          -0.053084628508673355196 * (round_tfidf_char_nblr_toxic <= 0.03526754304766655 and 
                                     round_tfidf_union_lr_toxic > 0.018717188388109207 and 
                                     round_tfidf_union_lr_insult <= 0.030532937496900558),
            0.04272927534158787316 * (round_tfidf_union_lr_toxic > 0.85472875833511353 and 
                                     round_tfidf_union_lr_insult > 0.030532937496900558 and 
                                     round_tfidf_union_nblr_toxic > 0.55363655090332031),
         -0.0082994848427264990287 * (round_tfidf_char_nblr_toxic > 0.03526754304766655 and 
                                     round_tfidf_union_lr_toxic <= 0.1849924772977829 and 
                                     round_tfidf_union_lr_insult <= 0.030532937496900558),
           0.020568581144857579768 * (round_tfidf_word_stop_nblr_toxic > 0.059217773377895355 and 
                                     round_tfidf_word_nostop_nblr_toxic > 0.017241643741726875 and 
                                     round_tfidf_union_nblr_toxic <= 0.1971437931060791),
           0.032076793162609439081 * (round_tfidf_char_nblr_obscene <= 0.062844991683959961 and 
                                     round_tfidf_union_lr_toxic <= 0.62230426073074341 and 
                                     round_tfidf_union_nblr_toxic > 0.1971437931060791),
           0.025173885457003192245 * (round_tfidf_char_nblr_obscene > 0.062844991683959961 and 
                                     round_tfidf_union_lr_toxic <= 0.62230426073074341 and 
                                     round_tfidf_union_nblr_toxic > 0.1971437931060791),
          0.0091635339449286992708 * (round_tfidf_word_stop_nblr_toxic <= 0.9828227162361145 and 
                                     round_tfidf_union_lr_toxic > 0.62230426073074341 and 
                                     round_tfidf_union_nblr_toxic > 0.1971437931060791),
            0.14206053491855924475 * (round_tfidf_word_stop_nblr_toxic > 0.9828227162361145 and 
                                     round_tfidf_union_lr_toxic > 0.62230426073074341 and 
                                     round_tfidf_union_nblr_toxic > 0.1971437931060791),
          -0.069982191045453431832 * (round_tfidf_word_stop_nblr_toxic <= 0.026927132159471512 and 
                                     round_tfidf_char_lr_toxic <= 0.13766011595726013 and 
                                     round_tfidf_union_nblr_toxic <= 0.011788342148065567),
            0.01148664875440251476 * (round_tfidf_word_stop_nblr_toxic <= 0.90966320037841797 and 
                                     round_tfidf_char_lr_toxic > 0.13766011595726013 and 
                                     round_tfidf_char_nblr_toxic > 0.49179011583328247),
           0.030762482924662454958 * (round_tfidf_word_stop_nblr_toxic > 0.90966320037841797 and 
                                     round_tfidf_char_lr_toxic > 0.13766011595726013 and 
                                     round_tfidf_char_nblr_toxic > 0.49179011583328247),
           -0.10504921354544639878 * (round_tfidf_word_stop_nblr_toxic <= 0.013227010145783424 and 
                                     round_tfidf_char_lr_toxic <= 0.044812910258769989 and 
                                     round_tfidf_char_nblr_toxic <= 0.13057276606559753),
           -0.05283553806008086523 * (round_tfidf_word_stop_nblr_toxic > 0.013227010145783424 and 
                                     round_tfidf_char_lr_toxic <= 0.044812910258769989 and 
                                     round_tfidf_char_nblr_toxic <= 0.13057276606559753),
          -0.035071144730845785209 * (round_tfidf_word_stop_lr_toxic <= 0.11656874418258667 and 
                                     round_tfidf_char_lr_toxic > 0.044812910258769989 and 
                                     round_tfidf_char_nblr_toxic <= 0.13057276606559753),
           -0.04606126609662784116 * (round_tfidf_char_lr_toxic > 0.036521986126899719 and 
                                     round_tfidf_char_nblr_toxic <= 0.049085929989814758 and 
                                     round_tfidf_union_nblr_toxic <= 0.084263712167739868),
         -0.0025856107755556603962 * (round_tfidf_char_lr_toxic > 0.036521986126899719 and 
                                     round_tfidf_char_nblr_toxic > 0.049085929989814758 and 
                                     round_tfidf_union_nblr_toxic <= 0.084263712167739868),
           0.039896643654192279205 * (0.44963276386260986 < round_tfidf_char_nblr_toxic <= 0.8915334939956665 and 
                                     round_tfidf_union_nblr_toxic > 0.084263712167739868),
           0.074940665851495766314 * (round_tfidf_char_nblr_toxic > 0.8915334939956665 and 
                                     round_tfidf_union_nblr_toxic > 0.084263712167739868),
          -0.074695161423902756148 * (round_tfidf_char_lr_toxic <= 0.036521986126899719 and 
                                     round_tfidf_union_nblr_toxic <= 0.012097699567675591),
            0.01427317700120230709 * (round_tfidf_char_nblr_toxic <= 0.75636398792266846 and 
                                     round_tfidf_union_lr_toxic > 0.41643568873405457 and 
                                     round_tfidf_union_nblr_toxic > 0.097102269530296326),
           -0.05583489307112177169 * (round_tfidf_char_nblr_toxic <= 0.012969900853931904 and 
                                     round_tfidf_union_nblr_toxic <= 0.017488252371549606),
           0.020105920067893415931 * (round_tfidf_char_nblr_obscene > 0.02529611811041832 and 
                                     round_tfidf_union_lr_toxic <= 0.41643568873405457 and 
                                     round_tfidf_union_nblr_toxic > 0.097102269530296326),
           0.043583463494970720031 * (round_tfidf_char_nblr_toxic > 0.44023534655570984 and 
                                     0.084263712167739868 < round_tfidf_union_nblr_toxic <= 0.95791184902191162),
           0.040623175430146145348 * (round_tfidf_char_nblr_toxic > 0.44023534655570984 and 
                                     round_tfidf_union_nblr_toxic > 0.95791184902191162),
           0.030932918993385023765 * (round_tfidf_word_stop_nblr_toxic > 0.18815049529075623 and 
                                     round_tfidf_char_nblr_toxic <= 0.44023534655570984 and 
                                     round_tfidf_union_nblr_toxic > 0.084263712167739868),
          -0.097925385258233674235 * (round_tfidf_char_nblr_toxic <= 0.026479119434952736 and 
                                     round_tfidf_union_lr_toxic > 0.01646161824464798 and 
                                     round_tfidf_union_nblr_toxic <= 0.084263712167739868),
          -0.061695202201466729786 * (round_tfidf_word_nostop_nblr_toxic <= 0.04905947670340538 and 
                                     round_tfidf_char_nblr_toxic > 0.026479119434952736 and 
                                     round_tfidf_union_nblr_toxic <= 0.084263712167739868),
            -0.0201631613367600597 * (round_tfidf_char_nblr_toxic <= 0.079931028187274933 and 
                                     0.025995738804340363 < round_tfidf_union_nblr_toxic <= 0.1971437931060791),
           0.005005571670439312032 * (round_tfidf_word_stop_nblr_toxic <= 0.64869987964630127 and 
                                     round_tfidf_char_nblr_obscene > 0.029308849945664406 and 
                                     round_tfidf_union_nblr_toxic > 0.1971437931060791),
            0.01089853031714350072 * (round_tfidf_char_lr_toxic <= 0.13766011595726013 and 
                                     round_tfidf_char_nblr_toxic > 0.071452602744102478),
           0.042891253275378786458 * (round_tfidf_char_lr_toxic > 0.13766011595726013 and 
                                     round_tfidf_char_nblr_toxic > 0.58039271831512451 and 
                                     round_tfidf_union_lr_obscene <= 0.40888684988021851),
          -0.068326491098942732094 * (round_tfidf_word_nostop_nblr_toxic <= 0.018668511882424355 and 
                                     round_tfidf_char_nblr_toxic <= 0.021390561014413834 and 
                                     round_tfidf_union_nblr_toxic <= 0.13555708527565002),
          -0.015405046024189522277 * (round_tfidf_word_nostop_nblr_toxic > 0.018668511882424355 and 
                                     round_tfidf_char_nblr_toxic <= 0.049078319221735001 and 
                                     round_tfidf_union_nblr_toxic <= 0.13555708527565002),
          -0.091430419112677824289 * (round_tfidf_word_stop_nblr_toxic <= 0.045997567474842072 and 
                                     round_tfidf_union_lr_insult <= 0.00675920769572258 and 
                                     round_tfidf_union_nblr_insult <= 0.022974647581577301),
           0.022849460660105225945 * (round_tfidf_word_stop_nblr_toxic <= 0.10491146147251129 and 
                                     round_tfidf_union_nblr_insult > 0.022974647581577301),
          -0.062099810547864793542 * (round_tfidf_word_stop_nblr_toxic <= 0.045997567474842072 and 
                                     round_tfidf_char_nblr_toxic <= 0.033914782106876373 and 
                                     round_tfidf_union_lr_insult > 0.00675920769572258 and 
                                     round_tfidf_union_nblr_insult <= 0.022974647581577301),
          -0.028072206743810015905 * (round_tfidf_word_stop_nblr_toxic <= 0.045997567474842072 and 
                                     round_tfidf_char_nblr_toxic > 0.033914782106876373 and 
                                     round_tfidf_union_lr_insult > 0.00675920769572258 and 
                                     round_tfidf_union_nblr_insult <= 0.022974647581577301),
           0.013747214889246038572 * (round_tfidf_word_stop_lr_toxic > 0.1343916654586792 and 
                                     round_tfidf_char_nblr_toxic > 0.80137038230895996),
           -0.10377770940967404045 * (round_tfidf_word_stop_lr_toxic <= 0.1343916654586792 and 
                                     round_tfidf_word_stop_nblr_toxic <= 0.013491391204297543 and 
                                     round_tfidf_union_nblr_toxic <= 0.024931274354457855),
           0.013559885705027932556 * (round_tfidf_word_stop_lr_toxic <= 0.1343916654586792 and 
                                     round_tfidf_union_lr_toxic > 0.12861226499080658 and 
                                     round_tfidf_union_nblr_toxic > 0.024931274354457855),
         -0.0034678987304380712135 * (round_tfidf_word_stop_lr_toxic > 0.1343916654586792 and 
                                     round_tfidf_word_nostop_nblr_toxic <= 0.084395341575145721 and 
                                     round_tfidf_char_nblr_toxic <= 0.3949882984161377),
           0.028916818128759003842 * (0.10493162274360657 < round_tfidf_word_stop_nblr_toxic <= 0.65276813507080078 and 
                                     round_tfidf_char_nblr_toxic <= 0.31551635265350342),
          -0.068023209563919467824 * (round_tfidf_word_stop_nblr_toxic <= 0.10493162274360657 and 
                                     0.0066804494708776474 < round_tfidf_union_nblr_toxic <= 0.02639315277338028),
           0.055629854687123919676 * (round_tfidf_word_stop_nblr_toxic > 0.65276813507080078 and 
                                     round_tfidf_char_lr_obscene <= 0.63151144981384277),
           0.088886021748749308169 * (round_tfidf_word_stop_nblr_toxic > 0.65276813507080078 and 
                                     round_tfidf_char_lr_obscene > 0.63151144981384277),
            0.02728416208777930535 * (round_tfidf_word_nostop_lr_toxic <= 0.12936381995677948 and 
                                     round_tfidf_char_nblr_toxic > 0.030666567385196686 and 
                                     round_tfidf_union_nblr_obscene <= 0.017556861042976379),
           0.029547849907591326041 * (round_tfidf_word_nostop_lr_toxic <= 0.12936381995677948 and 
                                     round_tfidf_char_nblr_toxic > 0.030666567385196686 and 
                                     round_tfidf_union_nblr_obscene > 0.017556861042976379),
           0.026931359918129685327 * (round_tfidf_word_nostop_lr_toxic > 0.12936381995677948 and 
                                     round_tfidf_union_lr_obscene <= 0.4832877516746521 and 
                                     round_tfidf_union_nblr_toxic > 0.67413699626922607),
           -0.11215210606937837945 * (round_tfidf_char_nblr_toxic <= 0.025055509060621262),
           0.084805766952560243666 * (round_tfidf_word_stop_nblr_toxic <= 0.65276813507080078 and 
                                     round_tfidf_char_nblr_toxic > 0.31930980086326599),
           0.025366739354250034816 * (round_tfidf_word_stop_nblr_toxic > 0.65276813507080078 and 
                                     round_tfidf_char_lr_severe_toxic > 0.019715569913387299 and 
                                     round_tfidf_char_nblr_toxic > 0.13057276606559753),
           0.010664223028089942283 * (round_tfidf_word_stop_nblr_toxic <= 0.19097228348255157 and 
                                     round_tfidf_char_lr_toxic > 0.13766011595726013 and 
                                     round_tfidf_union_nblr_toxic <= 0.60066473484039307),
           -0.42864534962549982744 * (round_tfidf_char_lr_toxic <= 0.13766011595726013 and 
                                     round_tfidf_union_lr_toxic <= 0.016685452312231064 and 
                                     round_tfidf_union_nblr_toxic <= 0.033640481531620026),
           -0.12495491077875012231 * (round_tfidf_char_lr_toxic <= 0.13766011595726013 and 
                                     round_tfidf_union_lr_toxic > 0.016685452312231064 and 
                                     round_tfidf_union_nblr_toxic <= 0.033640481531620026),
          -0.021538119310272756807 * (round_tfidf_word_nostop_nblr_toxic <= 0.072724722325801849 and 
                                     round_tfidf_char_lr_toxic <= 0.13766011595726013 and 
                                     round_tfidf_union_nblr_toxic > 0.033640481531620026),
           0.050220662115117990776 * (round_tfidf_char_lr_toxic > 0.13766011595726013 and 
                                     round_tfidf_char_lr_obscene <= 0.21254833042621613 and 
                                     round_tfidf_union_nblr_toxic > 0.60066473484039307),
           0.026872814073256545003 * (round_tfidf_word_stop_nblr_toxic > 0.078372091054916382 and 
                                     round_tfidf_word_nostop_lr_obscene <= 0.10476373881101608 and 
                                     round_tfidf_char_nblr_insult > 0.049090005457401276),
           0.061437443186346391011 * (round_tfidf_char_nblr_toxic > 0.13917550444602966 and 
                                     round_tfidf_union_lr_obscene <= 0.10073898732662201 and 
                                     round_tfidf_union_nblr_toxic <= 0.44755065441131592),
           0.061489281236116466456 * (round_tfidf_char_nblr_toxic > 0.13917550444602966 and 
                                     round_tfidf_union_lr_obscene <= 0.10073898732662201 and 
                                     round_tfidf_union_nblr_toxic > 0.44755065441131592),
          0.0082259638975297599134 * (round_tfidf_char_lr_obscene <= 0.64468848705291748 and 
                                     round_tfidf_char_nblr_toxic > 0.13917550444602966 and 
                                     round_tfidf_union_lr_obscene > 0.10073898732662201),
          -0.068294662942068540268 * (round_tfidf_word_nostop_nblr_toxic <= 0.01481745857745409 and 
                                     round_tfidf_char_nblr_toxic <= 0.023431099951267242),
           0.045233923444949308834 * (round_tfidf_word_stop_nblr_toxic <= 0.30503559112548828 and 
                                     round_tfidf_union_lr_toxic > 0.14234492182731628 and 
                                     round_tfidf_union_nblr_obscene <= 0.11884520947933197),
           0.045623062726051606675 * (round_tfidf_word_stop_nblr_toxic > 0.30503559112548828 and 
                                     round_tfidf_union_lr_toxic > 0.14234492182731628 and 
                                     round_tfidf_union_nblr_obscene <= 0.11884520947933197),
           -0.22209909093276306025 * (round_tfidf_char_nblr_toxic <= 0.032444044947624207 and 
                                     round_tfidf_union_lr_toxic <= 0.022474717348814011),
           0.036208030296950820259 * (round_tfidf_char_nblr_toxic > 0.032444044947624207 and 
                                     round_tfidf_char_nblr_obscene > 0.023999184370040894 and 
                                     round_tfidf_union_lr_toxic <= 0.14234492182731628),
            0.14649373372779664337 * (round_tfidf_char_nblr_toxic > 0.7010113000869751),
           -0.17250746536824249078 * (round_tfidf_word_nostop_lr_toxic <= 0.061595089733600616 and 
                                     round_tfidf_char_nblr_toxic <= 0.012084754183888435),
           0.046370798901339209397 * (round_tfidf_word_stop_nblr_toxic > 0.28250163793563843 and 
                                     0.11283569037914276 < round_tfidf_char_nblr_toxic <= 0.7010113000869751),
          -0.055131573778737223701 * (round_tfidf_word_stop_nblr_toxic <= 0.087411567568778992 and 
                                     round_tfidf_word_nostop_lr_toxic > 0.061595089733600616 and 
                                     round_tfidf_char_nblr_toxic <= 0.11283569037914276),
          -0.076371836344652430095 * (round_tfidf_word_nostop_lr_toxic <= 0.061595089733600616 and 
                                     0.012084754183888435 < round_tfidf_char_nblr_toxic <= 0.11283569037914276 and 
                                     round_num_capital <= 34.5),
           0.030080544619006877732 * (round_tfidf_word_stop_nblr_insult <= 0.014502024278044701 and 
                                     round_tfidf_char_nblr_obscene <= 0.0070003978908061981 and 
                                     round_tfidf_union_nblr_toxic > 0.033707432448863983),
          0.0024011943887945483636 * (round_tfidf_word_stop_nblr_toxic <= 0.10974167287349701 and 
                                     round_tfidf_word_stop_nblr_insult <= 0.014502024278044701 and 
                                     round_tfidf_char_nblr_obscene > 0.0070003978908061981),
           0.032187684304160697757 * (round_tfidf_word_stop_nblr_toxic > 0.10974167287349701 and 
                                     round_tfidf_word_stop_nblr_insult <= 0.014502024278044701 and 
                                     round_tfidf_char_nblr_obscene > 0.0070003978908061981),
          -0.049753633894272036864 * (round_tfidf_word_stop_nblr_insult > 0.014502024278044701 and 
                                     round_tfidf_word_nostop_nblr_insult <= 0.048280522227287292 and 
                                     round_tfidf_char_nblr_toxic <= 0.56561422348022461),
           0.039750903745875365569 * (round_tfidf_char_nblr_toxic > 0.17153346538543701 and 
                                     round_tfidf_union_lr_toxic > 0.7255634069442749),
          0.0060357361684728423265 * (round_tfidf_char_nblr_toxic <= 0.17153346538543701 and 
                                     round_tfidf_union_lr_toxic > 0.04427386075258255 and 
                                     round_capital_per_char > 0.2403007447719574),
             -0.086426901720318694 * (round_tfidf_word_nostop_nblr_toxic <= 0.019040808081626892 and 
                                     round_tfidf_char_nblr_toxic <= 0.17153346538543701 and 
                                     round_tfidf_union_lr_toxic <= 0.04427386075258255),
          -0.037072040692268562467 * (round_tfidf_word_nostop_nblr_toxic > 0.019040808081626892 and 
                                     round_tfidf_char_nblr_toxic <= 0.17153346538543701 and 
                                     round_tfidf_union_lr_toxic <= 0.04427386075258255),
           0.012281648179806807508 * (round_tfidf_word_stop_nblr_obscene <= 0.061478473246097565 and 
                                     round_tfidf_char_nblr_toxic > 0.17153346538543701 and 
                                     round_tfidf_union_lr_toxic <= 0.7255634069442749),
          -0.063527122939697200366 * (round_tfidf_word_nostop_nblr_toxic <= 0.033049892634153366 and 
                                     round_tfidf_union_lr_insult <= 0.043021798133850098 and 
                                     round_tfidf_union_nblr_insult <= 0.0053769247606396675 and 
                                     round_capital_per_char <= 0.32712829113006592),
          -0.028414935913989785438 * (round_tfidf_word_nostop_nblr_toxic > 0.033049892634153366 and 
                                     round_tfidf_word_nostop_nblr_obscene <= 0.022467408329248428 and 
                                     round_tfidf_union_lr_insult <= 0.043021798133850098),
           -0.15703250326859882757 * (round_tfidf_word_stop_nblr_toxic <= 0.017298439517617226 and 
                                     round_tfidf_word_stop_nblr_obscene <= 0.014564808458089828 and 
                                     round_tfidf_char_nblr_toxic <= 0.065406925976276398),
           0.011932094271030783614 * (round_tfidf_word_stop_nblr_toxic > 0.25399470329284668 and 
                                     round_tfidf_word_stop_nblr_obscene <= 0.014564808458089828 and 
                                     round_tfidf_char_nblr_toxic > 0.065406925976276398),
          -0.060571617686056698771 * (round_tfidf_word_stop_nblr_toxic > 0.017298439517617226 and 
                                     round_tfidf_word_stop_nblr_obscene <= 0.014564808458089828 and 
                                     round_tfidf_char_nblr_toxic <= 0.065406925976276398 and 
                                     round_lowercase_per_char > 0.098457790911197662),
          0.0097467308133070695852 * (round_tfidf_char_lr_obscene <= 0.13787031173706055 and 
                                     round_tfidf_union_nblr_toxic > 0.27280494570732117),
           -0.10196820835066223998 * (round_tfidf_word_nostop_nblr_toxic <= 0.029262559488415718 and 
                                     round_tfidf_union_lr_toxic <= 0.04574565589427948 and 
                                     round_tfidf_union_nblr_toxic <= 0.27280494570732117),
          -0.049979922676927607261 * (round_tfidf_word_nostop_nblr_toxic <= 0.029262559488415718 and 
                                     round_tfidf_union_lr_toxic > 0.04574565589427948 and 
                                     round_tfidf_union_nblr_toxic <= 0.27280494570732117),
          -0.015784519207097641447 * (round_tfidf_word_nostop_nblr_toxic > 0.029262559488415718 and 
                                     round_tfidf_char_nblr_obscene <= 0.023888709023594856 and 
                                     round_tfidf_union_nblr_toxic <= 0.27280494570732117 and 
                                     round_num_words_lower > 0.5),
           0.049234128588220564537 * (round_tfidf_word_stop_lr_obscene <= 0.078831747174263 and 
                                     round_tfidf_word_nostop_nblr_toxic > 0.084685996174812317),
           0.024480044485422035316 * (round_tfidf_word_nostop_nblr_toxic <= 0.084685996174812317 and 
                                     round_tfidf_char_nblr_toxic > 0.091050192713737488 and 
                                     round_tfidf_char_nblr_insult > 0.0080856764689087868),
           0.034912781368237277846 * (0.078831747174263 < round_tfidf_word_stop_lr_obscene <= 0.31759428977966309 and 
                                     round_tfidf_word_nostop_nblr_toxic > 0.084685996174812317),
          -0.089204420001654211037 * (round_tfidf_word_nostop_nblr_toxic <= 0.084685996174812317 and 
                                     round_tfidf_char_nblr_toxic <= 0.015608685091137886 and 
                                     round_tfidf_char_nblr_insult <= 0.0080856764689087868 and 
                                     round_capital_per_char <= 0.28222167491912842),
           0.047219594598069153379 * (round_tfidf_char_lr_severe_toxic > 0.016083888709545135 and 
                                     round_tfidf_union_nblr_toxic > 0.75778090953826904),
          -0.056828569479438900691 * (round_tfidf_word_nostop_nblr_obscene <= 0.0046752290800213814 and 
                                     round_tfidf_union_nblr_toxic <= 0.038582578301429749 and 
                                     round_capital_per_char <= 0.48863637447357178),
           -0.30903110801643612904 * (round_tfidf_word_stop_nblr_toxic <= 0.1441364586353302 and 
                                     round_tfidf_char_nblr_toxic <= 0.045027494430541992 and 
                                     round_tfidf_union_nblr_toxic <= 0.0080320965498685837 and 
                                     round_lowercase_per_char > 0.13270440697669983),
            -0.2250330667155022446 * (round_tfidf_word_stop_nblr_toxic <= 0.1441364586353302 and 
                                     round_tfidf_char_nblr_toxic <= 0.045027494430541992 and 
                                     round_tfidf_union_nblr_toxic > 0.0080320965498685837 and 
                                     round_lowercase_per_char > 0.13270440697669983),
          -0.019586037080404395533 * (round_tfidf_word_stop_nblr_toxic > 0.1441364586353302 and 
                                     round_tfidf_char_lr_severe_toxic <= 0.034268971532583237 and 
                                     round_tfidf_char_nblr_toxic > 0.70727765560150146),
          -0.032077741620691171964 * (round_tfidf_word_stop_nblr_toxic <= 0.1441364586353302 and 
                                     round_tfidf_char_nblr_toxic > 0.045027494430541992 and 
                                     round_tfidf_char_nblr_insult <= 0.041473925113677979),
          0.0013153633775348162922 * (round_tfidf_word_nostop_nblr_toxic > 0.084395341575145721 and 
                                     round_tfidf_char_lr_obscene > 0.10636001825332642),
          -0.033684188016407330679 * (round_tfidf_word_nostop_nblr_toxic <= 0.084395341575145721 and 
                                     round_tfidf_union_nblr_toxic > 0.033178701996803284 and 
                                     round_capital_per_char <= 0.26733142137527466),
           0.047601879872579216935 * (round_tfidf_word_stop_nblr_toxic > 0.54121756553649902 and 
                                     round_tfidf_word_nostop_nblr_toxic > 0.084395341575145721 and 
                                     round_tfidf_char_lr_obscene <= 0.10636001825332642 and 
                                     round_lowercase_per_char > 0.29625549912452698),
          -0.034857645663430220084 * (round_tfidf_word_stop_nblr_toxic > 0.022245321422815323 and 
                                     round_tfidf_char_nblr_toxic <= 0.079653501510620117 and 
                                     round_tfidf_union_lr_toxic <= 0.22459940612316132 and 
                                     round_lowercase_per_char > 0.12699821591377258),
           0.030454886557509015105 * (round_tfidf_union_lr_toxic > 0.22459940612316132 and 
                                     round_tfidf_union_lr_obscene <= 0.23131497204303741 and 
                                     round_tfidf_union_nblr_insult <= 0.15091356635093689),
            0.01165131069221715665 * (round_tfidf_union_lr_toxic > 0.22459940612316132 and 
                                     round_tfidf_union_lr_obscene <= 0.23131497204303741 and 
                                     round_tfidf_union_nblr_insult > 0.15091356635093689),
          -0.010286686183682655638 * (round_tfidf_char_nblr_insult <= 0.046227984130382538 and 
                                     round_tfidf_union_lr_toxic > 0.044437237083911896 and 
                                     round_tfidf_union_nblr_toxic <= 0.27280494570732117 and 
                                     round_capital_per_char <= 0.40603798627853394),
           0.064576648577529954798 * (round_tfidf_word_stop_lr_insult <= 0.18071554601192474 and 
                                     round_tfidf_union_nblr_toxic > 0.27280494570732117 and 
                                     round_num_lowercase <= 277.5),
          -0.079185083088114394112 * (round_tfidf_word_stop_nblr_toxic <= 0.12852446734905243 and 
                                     round_tfidf_char_nblr_toxic > 0.045027494430541992 and 
                                     round_capital_per_char <= 0.2403007447719574),
          0.0043455286978003829917 * (round_tfidf_word_nostop_lr_severe_toxic <= 0.012737400829792023 and 
                                     round_tfidf_char_nblr_toxic > 0.19095213711261749 and 
                                     round_tfidf_union_nblr_toxic <= 0.73920369148254395),
           0.050521561450833918361 * (round_tfidf_char_nblr_toxic > 0.17153346538543701 and 
                                     round_tfidf_union_nblr_toxic <= 0.76821696758270264),
           0.046949244442047337322 * (round_tfidf_char_nblr_toxic > 0.17153346538543701 and 
                                     round_tfidf_union_nblr_toxic > 0.76821696758270264 and 
                                     round_num_lowercase <= 218.5),
           -0.06026951739344230119 * (round_tfidf_word_nostop_nblr_toxic <= 0.02401852048933506 and 
                                     round_tfidf_char_nblr_toxic <= 0.079660527408123016 and 
                                     round_num_words_upper <= 9.5),
         -0.0038122729307085365114 * (round_tfidf_word_nostop_nblr_toxic > 0.02401852048933506 and 
                                     round_tfidf_char_nblr_toxic <= 0.079660527408123016 and 
                                     round_num_words_upper <= 9.5),
           0.017545253097149899474 * (round_tfidf_word_stop_nblr_insult > 0.069836653769016266 and 
                                     round_tfidf_char_nblr_toxic > 0.079660527408123016 and 
                                     round_tfidf_union_nblr_toxic <= 0.73119103908538818),
           0.057447413150260702441 * (round_tfidf_char_lr_severe_toxic <= 0.014827052131295204 and 
                                     round_tfidf_char_nblr_toxic > 0.079660527408123016 and 
                                     round_tfidf_union_nblr_toxic > 0.73119103908538818),
           0.080402375312711882316 * (round_tfidf_char_lr_severe_toxic > 0.014827052131295204 and 
                                     round_tfidf_char_nblr_toxic > 0.079660527408123016 and 
                                     round_tfidf_union_nblr_toxic > 0.73119103908538818),
          0.0048756469988284964479 * (round_tfidf_word_stop_nblr_insult <= 0.069836653769016266 and 
                                     round_tfidf_char_nblr_toxic > 0.079660527408123016 and 
                                     round_tfidf_union_nblr_toxic <= 0.73119103908538818 and 
                                     round_num_words_lower <= 0.5),
          -0.077123488976195489486 * (round_tfidf_word_nostop_nblr_insult <= 0.0072794193401932716 and 
                                     round_tfidf_char_nblr_toxic <= 0.21901458501815796 and 
                                     round_tfidf_char_nblr_obscene <= 0.0049549685791134834 and 
                                     round_num_capital <= 114.5),
          -0.031516552291451449541 * (round_tfidf_word_nostop_nblr_insult <= 0.0072794193401932716 and 
                                     round_tfidf_char_nblr_toxic <= 0.21901458501815796 and 
                                     round_tfidf_char_nblr_obscene > 0.0049549685791134834 and 
                                     round_num_capital <= 114.5),
          0.0063338967250403943599 * (round_tfidf_word_nostop_lr_severe_toxic <= 0.011254288256168365 and 
                                     round_tfidf_char_nblr_toxic > 0.21901458501815796 and 
                                     round_num_words_lower <= 74.5),
          -0.073521247951573914792 * (round_tfidf_word_nostop_nblr_toxic <= 0.027802521362900734 and 
                                     round_tfidf_char_nblr_toxic <= 0.076988115906715393 and 
                                     round_num_words_upper <= 8.5),
          -0.038179916472646707737 * (round_tfidf_word_nostop_nblr_toxic > 0.027802521362900734 and 
                                     round_tfidf_char_nblr_toxic <= 0.076988115906715393 and 
                                     round_num_words_upper <= 8.5),
          -0.067968757158397868912 * (round_tfidf_char_nblr_toxic <= 0.22204595804214478 and 
                                     round_tfidf_char_nblr_insult <= 0.0046641035005450249),
          -0.029594564111202287249 * (round_tfidf_char_nblr_toxic <= 0.22204595804214478 and 
                                     round_tfidf_char_nblr_insult > 0.0046641035005450249 and 
                                     round_tfidf_union_nblr_obscene <= 0.016780851408839226 and 
                                     round_capital_per_char <= 0.18469899892807007),
           0.033822229749961872647 * (round_tfidf_word_nostop_lr_severe_toxic <= 0.010421276092529297 and 
                                     round_tfidf_char_nblr_toxic > 0.22204595804214478 and 
                                     round_num_lowercase <= 430.0),
           0.014927664896943467626 * (round_tfidf_word_nostop_lr_severe_toxic > 0.010421276092529297 and 
                                     0.22204595804214478 < round_tfidf_char_nblr_toxic <= 0.99848288297653198),
           -0.16886634515753906616 * (round_tfidf_word_nostop_nblr_toxic <= 0.045094914734363556 and 
                                     round_tfidf_char_nblr_insult <= 0.03194798156619072 and 
                                     round_tfidf_union_lr_toxic <= 0.031464032828807831 and 
                                     round_num_words_upper <= 26.5),
            0.12748301088551622362 * (round_tfidf_union_lr_severe_toxic > 0.028762020170688629 and 
                                     round_tfidf_union_nblr_toxic > 0.373676598072052),
          -0.098895863147628590428 * (round_tfidf_word_stop_nblr_toxic <= 0.0194883793592453 and 
                                     round_tfidf_union_lr_toxic <= 0.063176028430461884 and 
                                     round_tfidf_union_nblr_toxic <= 0.373676598072052),
          -0.017675045367541438596 * (round_tfidf_word_stop_nblr_toxic > 0.0194883793592453 and 
                                     round_tfidf_union_lr_toxic <= 0.063176028430461884 and 
                                     round_tfidf_union_nblr_toxic <= 0.373676598072052),
           0.031732842222405889532 * (round_tfidf_union_lr_severe_toxic <= 0.028762020170688629 and 
                                     round_tfidf_union_nblr_toxic > 0.373676598072052 and 
                                     round_num_unique_words <= 63.5),
          -0.091527023408325289267 * (round_tfidf_word_nostop_nblr_insult <= 0.011179415509104729 and 
                                     round_tfidf_char_nblr_toxic <= 0.046878300607204437 and 
                                     round_num_capital <= 165.5),
          0.0078176434539975608873 * (round_tfidf_word_stop_nblr_insult <= 0.11010251939296722 and 
                                     round_tfidf_word_nostop_nblr_obscene > 0.10469412803649902 and 
                                     round_tfidf_word_nostop_nblr_insult > 0.011179415509104729 and 
                                     round_lowercase_per_char > 0.27953487634658813),
           0.051977263741157138377 * (round_tfidf_word_stop_nblr_toxic > 0.068487614393234253 and 
                                     round_tfidf_word_stop_nblr_obscene > 0.073803216218948364 and 
                                     round_num_lowercase <= 436.5),
           0.015721417845089794701 * (round_tfidf_word_stop_nblr_toxic > 0.068487614393234253 and 
                                     round_tfidf_word_stop_nblr_obscene <= 0.073803216218948364 and 
                                     round_tfidf_char_nblr_insult > 0.049794286489486694),
         0.00056309282518902416259 * (round_tfidf_word_stop_nblr_toxic <= 0.15274231135845184 and 
                                     round_tfidf_union_nblr_obscene > 0.0072985501028597355),
            0.02912111358311081713 * (round_tfidf_word_stop_nblr_toxic > 0.15274231135845184 and 
                                     round_tfidf_word_nostop_lr_severe_toxic > 0.016049619764089584),
          -0.062764361185840361612 * (round_tfidf_word_stop_nblr_toxic <= 0.15274231135845184 and 
                                     round_tfidf_char_nblr_toxic <= 0.016122423112392426 and 
                                     round_tfidf_union_nblr_obscene <= 0.0072985501028597355 and 
                                     round_lowercase_per_char > 0.14913249015808105),
            0.02198033828141332871 * (round_tfidf_word_stop_lr_severe_toxic <= 0.011743555776774883 and 
                                     round_tfidf_word_nostop_nblr_insult > 0.021584164351224899),
           0.043696793118771108666 * (round_tfidf_word_stop_lr_severe_toxic > 0.011743555776774883 and 
                                     round_tfidf_word_nostop_nblr_insult > 0.021584164351224899),
           0.045896722155242035412 * (round_tfidf_word_stop_nblr_toxic > 0.068346299231052399 and 
                                     round_tfidf_word_stop_nblr_insult <= 0.065144337713718414 and 
                                     round_tfidf_union_nblr_obscene <= 0.15892603993415833),
            0.04999798732936556328 * (round_tfidf_word_stop_nblr_toxic > 0.068346299231052399 and 
                                     round_tfidf_word_stop_nblr_insult > 0.065144337713718414 and 
                                     round_tfidf_union_nblr_obscene <= 0.15892603993415833),
           0.021062271993042600005 * (round_tfidf_word_stop_nblr_toxic > 0.43045544624328613 and 
                                     round_tfidf_word_nostop_nblr_toxic > 0.052496999502182007 and 
                                     round_tfidf_word_nostop_nblr_obscene <= 0.27196460962295532),
           0.023186890613075723799 * (round_tfidf_word_stop_nblr_toxic > 0.03343142569065094 and 
                                     round_tfidf_char_nblr_obscene <= 0.013156208209693432 and 
                                     round_capital_per_char <= 0.37222224473953247),
           0.025849101209223603048 * (round_tfidf_word_nostop_nblr_obscene <= 0.65912461280822754 and 
                                     round_tfidf_char_nblr_obscene > 0.023888709023594856 and 
                                     round_num_stopwords <= 24.5),
          -0.085064013250952683065 * (round_tfidf_word_stop_nblr_toxic <= 0.023671997711062431 and 
                                     round_tfidf_char_nblr_obscene <= 0.023888709023594856 and 
                                     round_tfidf_union_nblr_insult <= 0.012167153880000114 and 
                                     round_capital_per_char <= 0.64434522390365601),
           0.042897898334476727844 * (0.16307151317596436 < round_tfidf_word_nostop_nblr_toxic <= 0.7196662425994873 and 
                                     round_num_words_lower <= 51.5),
           0.037239940340968544219 * (round_tfidf_word_nostop_nblr_toxic > 0.7196662425994873 and 
                                     round_num_words_lower <= 51.5),
           0.007640940597576723059 * (round_tfidf_word_nostop_nblr_toxic <= 0.16307151317596436 and 
                                     round_tfidf_char_nblr_obscene > 0.0076914657838642597 and 
                                     round_tfidf_char_nblr_insult > 0.077862657606601715 and 
                                     round_num_capital <= 55.5),
          -0.023405896363741188898 * (round_tfidf_word_stop_nblr_identity_hate <= 0.00066047313157469034 and 
                                     round_tfidf_word_nostop_nblr_toxic <= 0.16307151317596436 and 
                                     round_tfidf_char_nblr_obscene <= 0.0076914657838642597 and 
                                     round_num_capital <= 55.5),
            0.13274882938917448305 * (round_tfidf_union_lr_toxic > 0.90169239044189453 and 
                                     round_tfidf_union_nblr_insult > 0.0468711256980896 and 
                                     round_num_lowercase <= 224.5),
          -0.010961895081406334077 * (0.0056983539834618568 < round_tfidf_word_nostop_nblr_toxic <= 0.03735467791557312 and 
                                     round_tfidf_union_nblr_insult <= 0.0468711256980896 and 
                                     round_capital_per_char <= 0.18713521957397461),
           0.011707525372316923695 * (round_tfidf_union_lr_toxic <= 0.90169239044189453 and 
                                     round_tfidf_union_nblr_insult > 0.0468711256980896 and 
                                     round_num_lowercase <= 224.5 and 
                                     round_num_words_lower > 0.5),
          -0.037406979354127838089 * (round_tfidf_word_nostop_nblr_insult <= 0.011179415509104729 and 
                                     round_tfidf_union_lr_toxic <= 0.066362470388412476 and 
                                     round_num_capital <= 20.5),
           -0.06362947156020506978 * (round_tfidf_word_nostop_nblr_insult <= 0.011179415509104729 and 
                                     round_tfidf_union_lr_toxic <= 0.066362470388412476 and 
                                     20.5 < round_num_capital <= 207.5),
           0.047473838709518052792 * (round_tfidf_word_stop_nblr_toxic <= 0.89816415309906006 and 
                                     round_tfidf_char_nblr_toxic > 0.27908524870872498 and 
                                     round_num_words_lower <= 56.5),
            0.11587628566743227165 * (round_tfidf_word_stop_nblr_toxic > 0.89816415309906006 and 
                                     round_tfidf_char_nblr_toxic > 0.27908524870872498 and 
                                     round_num_words_lower <= 56.5),
           0.021986786326149653642 * (round_tfidf_word_stop_nblr_obscene > 0.020688379183411598 and 
                                     round_tfidf_word_nostop_nblr_obscene <= 0.30396375060081482 and 
                                     round_num_unique_words <= 82.5),
           0.047262532448198162072 * (round_tfidf_word_stop_nblr_obscene > 0.020688379183411598 and 
                                     round_tfidf_word_nostop_nblr_obscene > 0.30396375060081482 and 
                                     round_num_unique_words <= 82.5),
          -0.021524730743663952198 * (round_tfidf_char_nblr_toxic <= 0.069360055029392242 and 
                                     round_tfidf_union_nblr_obscene <= 0.032826274633407593 and 
                                     round_num_words_upper > 4.5),
          -0.078690363361338777626 * (round_tfidf_word_stop_nblr_identity_hate <= 0.00088893115753307939 and 
                                     round_tfidf_char_nblr_toxic <= 0.069360055029392242 and 
                                     round_tfidf_union_nblr_obscene <= 0.032826274633407593 and 
                                     round_num_words_upper <= 4.5),
          -0.019089249542200523169 * (round_tfidf_word_stop_nblr_identity_hate > 0.00088893115753307939 and 
                                     round_tfidf_char_nblr_toxic <= 0.069360055029392242 and 
                                     round_tfidf_union_nblr_obscene <= 0.032826274633407593 and 
                                     round_num_words_upper <= 4.5),
           0.021130100064264915188 * (round_tfidf_word_nostop_nblr_toxic > 0.6666109561920166 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic > 0.00086565484525635839 and 
                                     round_num_unique_words <= 85.5 and 
                                     round_capital_per_char <= 0.17316341400146484),
          -0.075748320387132014875 * (round_tfidf_word_nostop_nblr_toxic <= 0.034938529133796692 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic <= 0.00086565484525635839 and 
                                     round_num_chars <= 436.5),
            -0.1132734632102650324 * (round_tfidf_word_nostop_nblr_toxic <= 0.011778139509260654 and 
                                     round_tfidf_word_nostop_nblr_obscene <= 0.0098902899771928787 and 
                                     round_tfidf_char_nblr_insult <= 0.049794286489486694 and 
                                     round_capital_per_char <= 0.42928570508956909),
          -0.013122372267218245998 * (round_tfidf_word_nostop_nblr_toxic > 0.011778139509260654 and 
                                     round_tfidf_word_nostop_nblr_obscene <= 0.0098902899771928787 and 
                                     round_tfidf_char_nblr_insult <= 0.049794286489486694 and 
                                     round_capital_per_char <= 0.42928570508956909),
            0.10369334361988427617 * (round_tfidf_char_nblr_toxic > 0.97828876972198486 and 
                                     round_tfidf_char_nblr_insult > 0.049794286489486694 and 
                                     round_num_stopwords <= 24.5),
          -0.015657378153006981658 * (round_tfidf_word_stop_nblr_toxic > 0.038008801639080048 and 
                                     round_tfidf_word_stop_nblr_obscene <= 0.035109903663396835 and 
                                     round_tfidf_union_nblr_insult <= 0.098065905272960663 and 
                                     round_lowercase_per_char > 0.14860057830810547),
         -0.0063236306872401296109 * (round_tfidf_word_nostop_nblr_obscene <= 0.011438643559813499 and 
                                     round_tfidf_char_nblr_toxic <= 0.31613564491271973 and 
                                     round_tfidf_union_nblr_insult > 0.0061239665374159813 and 
                                     round_num_words_upper <= 12.5),
           0.035795280905467359578 * (round_tfidf_word_nostop_nblr_obscene <= 0.011438643559813499 and 
                                     round_tfidf_char_nblr_toxic > 0.31613564491271973 and 
                                     round_tfidf_union_nblr_insult > 0.0061239665374159813 and 
                                     round_num_words_upper <= 12.5),
           0.074907558914688004981 * (round_tfidf_word_nostop_nblr_toxic <= 0.25437971949577332 and 
                                     round_num_words_upper > 26.5),
          -0.020762595351443115937 * (round_tfidf_word_nostop_nblr_toxic <= 0.25437971949577332 and 
                                     round_tfidf_char_nblr_toxic <= 0.015958726406097412 and 
                                     round_num_words_upper <= 26.5),
           0.041598020781980042659 * (0.032826274633407593 < round_tfidf_union_nblr_obscene <= 0.85518157482147217 and 
                                     round_num_lowercase <= 493.0),
           0.025019173769842795868 * (round_tfidf_union_nblr_obscene > 0.85518157482147217 and 
                                     round_num_lowercase <= 493.0),
          0.0068850638650936367771 * (round_tfidf_word_nostop_nblr_insult <= 0.0049675637856125832 and 
                                     round_tfidf_union_lr_toxic > 0.061001904308795929 and 
                                     round_tfidf_union_nblr_obscene <= 0.032826274633407593),
          -0.036855901970728931205 * (round_tfidf_word_nostop_nblr_toxic <= 0.018945956602692604 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic <= 0.00086565484525635839 and 
                                     round_num_capital <= 33.5),
           0.040704782043515512491 * (round_tfidf_word_nostop_nblr_severe_toxic > 0.00086565484525635839 and 
                                     round_tfidf_char_lr_severe_toxic > 0.023769602179527283),
          -0.051997179516559487167 * (round_tfidf_word_nostop_nblr_severe_toxic <= 0.00086565484525635839 and 
                                     round_tfidf_char_nblr_toxic <= 0.069347873330116272 and 
                                     round_num_lowercase <= 159.5 and 
                                     round_capital_per_char <= 0.18200221657752991),
           0.037314274157093305084 * (round_tfidf_word_stop_nblr_toxic > 0.03343142569065094 and 
                                     round_tfidf_word_nostop_nblr_toxic > 0.55059248208999634 and 
                                     round_tfidf_char_nblr_insult <= 0.050052810460329056 and 
                                     round_num_capital <= 191.0),
          0.0038741982241022859118 * (round_tfidf_word_nostop_nblr_severe_toxic <= 0.00086565484525635839 and 
                                     round_tfidf_union_nblr_insult > 0.016945933923125267 and 
                                     round_num_capital <= 98.0),
          -0.080350786826408476804 * (round_tfidf_word_stop_nblr_insult <= 0.027696024626493454 and 
                                     round_tfidf_word_nostop_nblr_toxic <= 0.010798148810863495 and 
                                     round_tfidf_char_nblr_obscene <= 0.012863532640039921 and 
                                     round_capital_per_char <= 0.13931721448898315),
           0.061507028822378892463 * (round_tfidf_union_nblr_obscene <= 0.055605024099349976 and 
                                     round_num_capital > 55.5),
          0.0073199801019054600015 * (round_tfidf_word_nostop_nblr_obscene <= 0.8855825662612915 and 
                                     round_tfidf_union_nblr_obscene > 0.055605024099349976 and 
                                     round_num_chars <= 587.5),
          -0.059372617097456462099 * (round_tfidf_word_nostop_nblr_toxic <= 0.29383116960525513 and 
                                     round_tfidf_char_nblr_obscene <= 0.034206949174404144 and 
                                     round_tfidf_union_nblr_toxic > 0.0047249919734895229 and 
                                     round_num_words_upper <= 5.5),
            0.04865342821042309418 * (round_tfidf_char_nblr_obscene <= 0.034206949174404144 and 
                                     round_num_capital > 200.5),
          -0.013884477879582812071 * (round_tfidf_word_stop_nblr_identity_hate <= 0.00088317925110459328 and 
                                     round_tfidf_word_nostop_nblr_obscene <= 0.0098902899771928787 and 
                                     round_tfidf_union_nblr_insult <= 0.098522752523422241 and 
                                     round_num_words_upper <= 30.5),
           0.049094908934182564242 * (round_tfidf_word_stop_nblr_severe_toxic > 0.011335864663124084 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic > 0.001112273195758462),
           0.020676727430665280127 * (round_tfidf_word_nostop_nblr_severe_toxic <= 0.001112273195758462 and 
                                     round_tfidf_char_nblr_insult > 0.012350985780358315 and 
                                     round_tfidf_union_lr_threat <= 0.00088095571845769882 and 
                                     round_num_words_upper <= 19.5),
          -0.039075476198872256817 * (round_tfidf_word_nostop_nblr_severe_toxic <= 0.001112273195758462 and 
                                     round_tfidf_word_nostop_nblr_identity_hate <= 0.00058945105411112309 and 
                                     round_tfidf_char_nblr_insult <= 0.012350985780358315 and 
                                     round_num_words_upper <= 19.5),
           0.011921510558945075733 * (round_tfidf_word_nostop_nblr_severe_toxic <= 0.0052647152915596962 and 
                                     round_tfidf_word_nostop_nblr_insult > 0.012199936434626579 and 
                                     round_num_unique_words <= 111.5 and 
                                     round_capital_per_char <= 0.13918545842170715),
           0.060786341402117124999 * (round_tfidf_word_nostop_nblr_severe_toxic > 0.0052647152915596962 and 
                                     round_tfidf_word_nostop_nblr_insult > 0.012199936434626579 and 
                                     round_num_unique_words <= 111.5 and 
                                     round_capital_per_char <= 0.13918545842170715),
          -0.035462988213226555312 * (round_tfidf_word_nostop_nblr_toxic <= 0.029318306595087051 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic <= 0.001112273195758462 and 
                                     round_num_words <= 85.5 and 
                                     round_num_words_upper <= 19.5),
          0.0078816074547780525655 * (round_tfidf_word_nostop_nblr_severe_toxic > 0.001112273195758462 and 
                                     round_tfidf_char_lr_threat > 0.01270957849919796),
          -0.047193705213887486571 * (round_tfidf_word_nostop_nblr_toxic <= 0.02851366251707077 and 
                                     round_tfidf_word_nostop_nblr_identity_hate <= 0.00085464841686189175 and 
                                     round_tfidf_char_nblr_obscene <= 0.034206949174404144 and 
                                     round_num_capital <= 33.5),
           0.006766562219965457578 * (round_tfidf_char_nblr_obscene > 0.034206949174404144 and 
                                     round_tfidf_union_lr_obscene <= 0.95117765665054321 and 
                                     round_num_unique_words <= 80.5),
         -0.0086456885397223036394 * (round_tfidf_word_nostop_nblr_severe_toxic <= 0.0012919659493491054 and 
                                     round_tfidf_union_lr_threat > 0.00088110670913010836 and 
                                     round_tfidf_union_lr_insult > 0.015050999820232391 and 
                                     round_num_capital <= 200.5),
           0.024678112118984244644 * (round_tfidf_word_stop_nblr_severe_toxic > 0.011079364456236362 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic > 0.0012919659493491054),
            0.11356735570178370931 * (round_tfidf_union_nblr_obscene <= 0.1614178866147995 and 
                                     round_num_capital > 197.5),
           -0.18066940740049694525 * (round_tfidf_union_nblr_toxic <= 0.0046463049948215485 and 
                                     round_tfidf_union_nblr_obscene <= 0.1614178866147995 and 
                                     round_num_capital <= 197.5),
          -0.017995482031171381593 * (round_tfidf_word_nostop_nblr_insult <= 0.069699846208095551 and 
                                     round_tfidf_union_nblr_toxic > 0.0046463049948215485 and 
                                     round_tfidf_union_nblr_obscene <= 0.1614178866147995 and 
                                     round_num_capital <= 197.5),
           0.010117181693852840899 * (round_tfidf_word_stop_nblr_obscene > 0.38217067718505859 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic > 0.0006383087020367384 and 
                                     round_lowercase_per_char > 0.25957384705543518),
         -0.0048028875084915965271 * (round_tfidf_word_stop_nblr_insult <= 0.15128540992736816 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0018341583199799061 and 
                                     round_tfidf_union_nblr_obscene > 0.0048848660662770271 and 
                                     round_capital_per_char <= 0.12561987340450287),
          -0.010966678270747065929 * (round_tfidf_word_stop_nblr_insult <= 0.15128540992736816 and 
                                     round_tfidf_word_nostop_nblr_insult <= 0.0049034184776246548 and 
                                     round_tfidf_union_nblr_obscene <= 0.0048848660662770271 and 
                                     round_capital_per_char <= 0.12561987340450287),
           0.024522450847854735528 * (round_tfidf_char_nblr_threat > 0.023802135139703751 and 
                                     round_tfidf_union_nblr_obscene <= 0.15920644998550415),
          -0.044814518986211741847 * (round_tfidf_word_stop_nblr_insult <= 0.0084121860563755035 and 
                                     round_tfidf_char_nblr_threat <= 0.023802135139703751 and 
                                     round_tfidf_union_nblr_obscene <= 0.15920644998550415 and 
                                     round_lowercase_per_char > 0.11001099646091461),
          -0.042023537575751793161 * (round_tfidf_word_stop_nblr_insult > 0.0084121860563755035 and 
                                     round_tfidf_char_nblr_threat <= 0.023802135139703751 and 
                                     round_tfidf_union_nblr_obscene <= 0.15920644998550415 and 
                                     round_lowercase_per_char > 0.11001099646091461),
         -0.0070710114193690797035 * (round_tfidf_word_stop_nblr_threat <= 0.049018807709217072 and 
                                     round_tfidf_word_stop_nblr_insult <= 0.15976917743682861 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic > 0.00063763017533347011 and 
                                     round_capital_per_char <= 0.12561987340450287),
          -0.020723371857940920543 * (round_tfidf_word_nostop_nblr_severe_toxic <= 0.0012916005216538906 and 
                                     round_tfidf_char_nblr_toxic > 0.0066171172074973583 and 
                                     round_tfidf_union_lr_threat > 0.00072027923306450248 and 
                                     round_tfidf_union_nblr_toxic <= 0.49068659543991089),
         -0.0063077502836177152395 * (round_tfidf_char_nblr_obscene > 0.0043122284114360809 and 
                                     round_tfidf_union_lr_threat > 0.00088101229630410671 and 
                                     round_tfidf_union_nblr_insult <= 0.16138613224029541 and 
                                     round_capital_per_char <= 0.12561987340450287),
           0.065913583235632206003 * (round_capital_per_char > 0.12561987340450287 and 
                                     round_punctuation_per_char <= 0.15304248034954071),
          -0.037030890933177344759 * (round_tfidf_char_nblr_obscene <= 0.0043122284114360809 and 
                                     round_tfidf_union_nblr_insult <= 0.0035547064617276192 and 
                                     round_capital_per_char <= 0.12561987340450287),
           0.019129301750244999403 * (round_tfidf_word_nostop_nblr_severe_toxic <= 0.00064335973002016544 and 
                                     round_tfidf_char_lr_insult <= 0.19222581386566162 and 
                                     round_tfidf_char_nblr_threat <= 0.023802135139703751 and 
                                     round_num_capital <= 277.0),
           0.055679136955296519573 * (round_tfidf_word_nostop_nblr_severe_toxic > 0.00064335973002016544 and 
                                     round_tfidf_char_lr_insult <= 0.19222581386566162 and 
                                     round_tfidf_char_nblr_threat <= 0.023802135139703751 and 
                                     round_num_capital <= 277.0),
           0.054644703103900828245 * (round_tfidf_word_stop_nblr_insult <= 0.038705781102180481 and 
                                     round_num_capital > 227.5),
          -0.017519382737677838963 * (round_tfidf_word_stop_nblr_insult <= 0.038705781102180481 and 
                                     round_tfidf_union_nblr_obscene <= 0.056565560400485992 and 
                                     round_num_capital <= 227.5 and 
                                     round_punctuation_per_char <= 0.10896660387516022),
           0.010342724663872360918 * (round_tfidf_word_nostop_nblr_obscene <= 0.33989125490188599 and 
                                     round_tfidf_char_nblr_insult > 0.015161613002419472 and 
                                     round_capital_per_char <= 0.13918545842170715 and 
                                     round_num_stopwords <= 96.5),
           0.066052789799038308383 * (round_tfidf_word_stop_nblr_threat > 0.0050962101668119431 and 
                                     round_tfidf_union_nblr_obscene <= 0.1614178866147995 and 
                                     round_capital_per_char <= 0.71581029891967773),
           0.044820192736578279291 * (round_tfidf_word_stop_nblr_threat <= 0.0050962101668119431 and 
                                     round_tfidf_char_nblr_insult > 0.10124041140079498 and 
                                     round_tfidf_union_nblr_obscene <= 0.1614178866147995 and 
                                     round_capital_per_char <= 0.71581029891967773),
          -0.061342306863900644687 * (round_tfidf_word_stop_nblr_identity_hate <= 0.37718924880027771 and 
                                     round_tfidf_word_nostop_nblr_threat <= 0.0077917245216667652 and 
                                     round_tfidf_union_nblr_insult <= 0.15105085074901581 and 
                                     round_capital_per_char <= 0.72763633728027344),
         -0.0039461776487698350946 * (round_tfidf_char_lr_identity_hate <= 0.022959563881158829 and 
                                     round_tfidf_union_lr_threat > 0.00088110670913010836 and 
                                     round_tfidf_union_nblr_obscene > 0.0048848660662770271 and 
                                     round_capital_per_char <= 0.12561987340450287),
           0.059035870789955112448 * (round_tfidf_char_nblr_threat > 0.0095096193253993988 and 
                                     round_tfidf_union_nblr_insult <= 0.16105562448501587),
           0.018598673966327149837 * (round_tfidf_union_nblr_insult > 0.16105562448501587 and 
                                     round_num_lowercase <= 135.5),
          -0.019050741536441220814 * (round_tfidf_word_stop_nblr_severe_toxic <= 0.0019906326197087765 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0019694536458700895 and 
                                     round_tfidf_char_lr_insult <= 0.20650330185890198 and 
                                     round_tfidf_char_nblr_threat <= 0.023802135139703751),
           0.013277783460341814614 * (round_tfidf_word_nostop_lr_threat > 0.0017055284697562456 and 
                                     round_tfidf_word_nostop_nblr_insult > 0.0055868308991193771 and 
                                     round_num_unique_words <= 127.5 and 
                                     round_capital_per_char <= 0.13825462758541107),
           0.057671506584567780451 * (round_num_capital > 80.0 and 
                                     round_capital_per_char > 0.13825462758541107),
           0.030148747592328935946 * (round_tfidf_word_stop_nblr_obscene <= 0.4111684262752533 and 
                                     round_tfidf_char_nblr_obscene > 0.012751834467053413 and 
                                     round_capital_per_char <= 0.095299459993839264 and 
                                     round_num_words_title <= 10.5),
           0.031470763256855986689 * (round_tfidf_word_stop_nblr_obscene > 0.4111684262752533 and 
                                     round_tfidf_char_nblr_obscene > 0.012751834467053413 and 
                                     round_capital_per_char <= 0.095299459993839264 and 
                                     round_num_words_title <= 10.5),
          -0.035737699607133806279 * (round_tfidf_word_stop_nblr_insult <= 0.20126329362392426 and 
                                     round_tfidf_word_nostop_nblr_threat <= 0.0073946584016084671 and 
                                     round_tfidf_char_nblr_obscene <= 0.0073391953483223915 and 
                                     round_num_words_upper <= 47.5),
          -0.032367133369355684247 * (round_tfidf_word_nostop_nblr_severe_toxic <= 0.0010192588670179248 and 
                                     round_tfidf_char_lr_toxic <= 0.077143177390098572 and 
                                     round_tfidf_char_lr_threat > 0.0013562282547354698),
           0.022719322217098898015 * (round_tfidf_word_nostop_nblr_severe_toxic > 0.0010192588670179248 and 
                                     round_num_words_title <= 4.5),
            0.04815651248790251554 * (round_tfidf_word_stop_lr_toxic > 0.067245066165924072 and 
                                     round_tfidf_union_nblr_obscene > 0.16260519623756409 and 
                                     round_num_chars <= 895.5),
          -0.040595201078596140909 * (round_tfidf_word_stop_nblr_insult <= 0.028960436582565308 and 
                                     round_tfidf_word_nostop_nblr_identity_hate <= 0.00058601354248821735 and 
                                     round_tfidf_char_nblr_toxic <= 0.29663223028182983 and 
                                     round_num_capital <= 22.5),
           -0.02664276103850691732 * (round_tfidf_word_stop_nblr_insult <= 0.028960436582565308 and 
                                     round_tfidf_word_nostop_nblr_identity_hate > 0.00058601354248821735 and 
                                     round_tfidf_char_nblr_toxic <= 0.29663223028182983 and 
                                     round_num_capital <= 22.5),
          -0.049296522763072711526 * (round_tfidf_word_stop_nblr_obscene <= 0.38449406623840332 and 
                                     round_tfidf_word_nostop_lr_threat <= 0.0021260497160255909 and 
                                     round_tfidf_word_nostop_nblr_insult <= 0.0042191152460873127 and 
                                     round_tfidf_char_nblr_threat <= 0.024544186890125275),
          -0.013255319541972936798 * (round_tfidf_word_stop_nblr_obscene <= 0.38449406623840332 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0021260497160255909 and 
                                     round_tfidf_char_lr_identity_hate <= 0.02413000725209713 and 
                                     round_tfidf_char_nblr_threat <= 0.024544186890125275),
           0.020732145758311788797 * (round_tfidf_word_stop_nblr_threat <= 0.0050962101668119431 and 
                                     round_tfidf_word_nostop_nblr_obscene <= 0.34616324305534363 and 
                                     round_tfidf_char_lr_toxic > 0.010240871459245682 and 
                                     round_tfidf_char_lr_threat <= 0.0012129424139857292),
          -0.006517273682447420266 * (round_tfidf_word_stop_nblr_insult > 0.012372855097055435 and 
                                     round_tfidf_char_nblr_severe_toxic <= 0.0010259924456477165 and 
                                     round_tfidf_union_lr_threat > 0.00070557778235524893),
           0.028671242043561084856 * (round_tfidf_word_nostop_nblr_severe_toxic <= 0.00063763017533347011 and 
                                     round_tfidf_char_nblr_toxic > 0.0066018775105476379 and 
                                     round_num_words > 10.5 and 
                                     round_num_chars <= 4957.5),
         -0.0084599416166063826084 * (round_tfidf_word_stop_nblr_threat <= 0.059206277132034302 and 
                                     round_tfidf_char_nblr_toxic > 0.0091980807483196259 and 
                                     round_tfidf_union_lr_threat > 0.00071806705091148615 and 
                                     round_capital_per_char <= 0.16718578338623047),
          -0.015681951895036280731 * (round_tfidf_word_stop_nblr_insult <= 0.20126329362392426 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0018891540821641684 and 
                                     round_tfidf_char_nblr_threat <= 0.010321669280529022 and 
                                     round_lowercase_per_char <= 0.79114961624145508),
           0.021020519828327092343 * (round_tfidf_char_nblr_severe_toxic > 0.001026347978040576 and 
                                     round_num_words <= 86.5),
           0.025566198529523451177 * (round_tfidf_word_nostop_nblr_obscene <= 0.21142926812171936 and 
                                     round_tfidf_char_nblr_insult > 0.015161613002419472 and 
                                     round_num_capital <= 236.5 and 
                                     round_num_words_title <= 9.5),
          -0.012683521620166773522 * (round_tfidf_word_nostop_nblr_obscene <= 0.21142926812171936 and 
                                     round_tfidf_word_nostop_nblr_identity_hate <= 0.00062390812672674656 and 
                                     round_tfidf_char_nblr_insult <= 0.015161613002419472 and 
                                     round_num_capital <= 236.5),
          -0.082076546194834412717 * (round_tfidf_word_nostop_lr_threat <= 0.0021229998674243689 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic <= 0.0012916005216538906 and 
                                     round_tfidf_char_nblr_toxic <= 0.010103452950716019),
          -0.012998156767278968085 * (round_tfidf_word_stop_nblr_toxic <= 0.56561410427093506 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0021229998674243689 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic <= 0.0012916005216538906 and 
                                     round_num_chars <= 4405.5),
          -0.024762941943671783196 * (round_tfidf_word_stop_nblr_insult <= 0.45071235299110413 and 
                                     round_tfidf_char_nblr_threat <= 0.31164416670799255 and 
                                     round_capital_per_char <= 0.095299459993839264 and 
                                     round_lowercase_per_char <= 0.794303297996521),
           0.048153840456648659696 * (round_tfidf_word_nostop_lr_threat > 0.0018823629943653941 and 
                                     round_tfidf_word_nostop_nblr_identity_hate > 0.11807948350906372 and 
                                     round_tfidf_union_nblr_obscene <= 0.45338663458824158 and 
                                     round_capital_per_char <= 0.72830867767333984),
           -0.03377272211762295806 * (round_tfidf_word_stop_nblr_identity_hate <= 0.23458901047706604 and 
                                     round_tfidf_word_nostop_lr_threat <= 0.0020529758185148239 and 
                                     round_tfidf_word_nostop_nblr_insult <= 0.0022864777129143476 and 
                                     round_tfidf_char_lr_insult <= 0.26457333564758301),
           0.010047607967066505066 * (round_tfidf_word_nostop_lr_threat > 0.0016752001829445362 and 
                                     round_tfidf_char_nblr_insult > 0.015161601826548576 and 
                                     round_num_stopwords <= 26.5 and 
                                     round_punctuation_per_char > 0.0060882940888404846),
           0.012477239485269771627 * (round_tfidf_word_nostop_lr_threat > 0.0016752001829445362 and 
                                     round_tfidf_word_nostop_nblr_toxic > 0.31326913833618164 and 
                                     round_tfidf_char_nblr_insult <= 0.015161601826548576 and 
                                     round_punctuation_per_char > 0.0060882940888404846),
          -0.056422146937988131021 * (round_tfidf_word_nostop_lr_threat <= 0.0016752001829445362 and 
                                     round_tfidf_char_nblr_toxic <= 0.0066170208156108856 and 
                                     round_punctuation_per_char > 0.0060882940888404846),
          -0.023011523892558217314 * (round_tfidf_word_nostop_lr_threat <= 0.0016752001829445362 and 
                                     round_tfidf_char_lr_identity_hate > 0.0026020645163953304 and 
                                     round_tfidf_char_nblr_toxic > 0.0066170208156108856 and 
                                     round_punctuation_per_char > 0.0060882940888404846),
           0.021344190873524335195 * (round_tfidf_word_nostop_lr_toxic > 0.16585177183151245 and 
                                     round_tfidf_char_nblr_obscene > 0.052481040358543396 and 
                                     round_tfidf_char_nblr_threat <= 0.0064780907705426216 and 
                                     round_num_words_lower <= 37.5),
          -0.008887017724458911791 * (round_tfidf_char_lr_identity_hate > 0.0033741926308721304 and 
                                     round_tfidf_char_nblr_obscene <= 0.052481040358543396 and 
                                     round_tfidf_char_nblr_threat <= 0.0064780907705426216 and 
                                     round_punctuation_per_char <= 0.12498742341995239),
          -0.060324512913657202251 * (round_tfidf_char_nblr_obscene <= 0.052481040358543396 and 
                                     round_tfidf_char_nblr_threat <= 0.0064780907705426216 and 
                                     round_num_punctuations <= 59.5 and 
                                     round_punctuation_per_char > 0.12498742341995239),
             0.0314562057077786561 * (round_tfidf_word_nostop_nblr_severe_toxic > 0.00063763017533347011 and 
                                     round_num_words_lower <= 137.5),
         -0.0042595416014245268688 * (round_tfidf_word_stop_nblr_identity_hate <= 0.0065837823785841465 and 
                                     round_tfidf_char_lr_threat > 0.0012129489332437515 and 
                                     round_num_unique_words <= 157.5 and 
                                     round_capital_per_char <= 0.095299459993839264),
           0.037585211268503344872 * (round_num_unique_words <= 156.5 and 
                                     round_num_chars > 52.5 and 
                                     round_num_capital > 44.5 and 
                                     round_punctuation_per_char > 0.0060882940888404846),
          -0.039988075052081753502 * (round_tfidf_char_lr_identity_hate <= 0.0027105552144348621 and 
                                     round_tfidf_union_nblr_severe_toxic <= 0.050816874951124191 and 
                                     round_tfidf_union_nblr_obscene > 0.0032931528985500336 and 
                                     round_punctuation_per_char <= 0.17320513725280762),
           -0.13483799883456409896 * (round_tfidf_char_lr_identity_hate > 0.0027105552144348621 and 
                                     round_tfidf_union_nblr_severe_toxic <= 0.050816874951124191 and 
                                     round_tfidf_union_nblr_obscene > 0.0032931528985500336 and 
                                     round_punctuation_per_char <= 0.17320513725280762),
           -0.18053695985911352717 * (round_tfidf_union_nblr_severe_toxic <= 0.050816874951124191 and 
                                     round_tfidf_union_nblr_obscene <= 0.0032931528985500336 and 
                                     round_num_capital <= 605.5 and 
                                     round_punctuation_per_char <= 0.17320513725280762),
            0.05124964951780451583 * (round_tfidf_char_lr_identity_hate > 0.019863206893205643 and 
                                     round_num_words_title <= 9.5),
           0.056777671980251624706 * (round_punctuation_per_char <= 0.0060882940888404846),
           0.031858556819429319218 * (round_tfidf_char_lr_insult > 0.038721688091754913 and 
                                     round_tfidf_char_lr_identity_hate > 0.0033741926308721304 and 
                                     round_num_unique_words <= 113.5 and 
                                     round_punctuation_per_char > 0.0060882940888404846),
          0.0099238371295154552409 * (round_num_capital > 20.5 and 
                                     round_capital_per_char > 0.095299459993839264 and 
                                     round_lowercase_per_char <= 0.79991698265075684 and 
                                     round_punctuation_per_char <= 0.15273231267929077),
           0.014415664984471360244 * (round_tfidf_char_nblr_obscene <= 0.0073391953483223915 and 
                                     round_tfidf_union_lr_insult > 0.014704696834087372 and 
                                     round_num_chars <= 664.5),
             0.0132278881976826794 * (round_tfidf_word_stop_nblr_obscene > 0.0079639721661806107 and 
                                     round_tfidf_word_nostop_nblr_threat <= 0.0001207602908834815),
           0.046023832698661566731 * (round_tfidf_word_nostop_nblr_severe_toxic > 0.033181443810462952),
           0.006442242558851018347 * (round_num_unique_words <= 151.5 and 
                                     round_num_chars <= 2627.5 and 
                                     round_capital_per_char <= 0.094172582030296326 and 
                                     round_punctuation_per_char <= 0.10456414520740509),
           0.035382941363570147297 * (round_num_capital > 15.5 and 
                                     round_capital_per_char > 0.094172582030296326 and 
                                     round_punctuation_per_char <= 0.10456414520740509),
          -0.051751003139931923969 * (round_tfidf_word_nostop_nblr_toxic <= 0.20470252633094788 and 
                                     round_tfidf_char_lr_threat > 0.0017956516239792109 and 
                                     round_tfidf_char_nblr_obscene <= 0.090765781700611115 and 
                                     round_tfidf_char_nblr_threat <= 0.0064489347860217094),
           0.010117623481746055442 * (round_tfidf_char_lr_threat <= 0.0017956516239792109 and 
                                     round_tfidf_char_nblr_obscene <= 0.090765781700611115 and 
                                     round_tfidf_char_nblr_threat <= 0.0064489347860217094 and 
                                     round_tfidf_char_nblr_insult > 0.0027273972518742085),
          -0.032981131948723041436 * (round_tfidf_word_stop_nblr_toxic <= 0.15097188949584961 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic <= 0.0005595602560788393 and 
                                     round_tfidf_char_nblr_insult <= 0.029510168358683586 and 
                                     round_num_chars <= 300.5),
           0.023575222980600080147 * (round_tfidf_word_nostop_lr_threat > 0.0021260497160255909 and 
                                     round_tfidf_char_lr_threat <= 0.012682989239692688 and 
                                     round_tfidf_char_nblr_insult > 0.24515166878700256 and 
                                     round_num_words_title <= 198.5),
           0.030119824578281086114 * (round_num_chars > 52.5 and 
                                     round_lowercase_per_char <= 0.45729482173919678),
          -0.009357616102654184656 * (round_tfidf_word_stop_nblr_toxic <= 0.60610747337341309 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0020529758185148239 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic <= 0.0012916005216538906 and 
                                     round_tfidf_word_nostop_nblr_identity_hate <= 0.22171244025230408),
          -0.054507368331603467393 * (round_tfidf_word_stop_lr_identity_hate > 0.0036029499024152756 and 
                                     round_tfidf_word_stop_nblr_toxic <= 0.037477843463420868 and 
                                     round_lowercase_per_char <= 0.79097294807434082 and 
                                     round_num_words_title <= 198.5),
          0.0089840981198198999191 * (round_tfidf_word_stop_lr_identity_hate > 0.0036029499024152756 and 
                                     round_tfidf_word_stop_nblr_toxic > 0.037477843463420868 and 
                                     round_lowercase_per_char <= 0.79097294807434082 and 
                                     round_num_words_title <= 198.5),
          -0.034928758192247520709 * (round_tfidf_word_stop_nblr_identity_hate <= 0.0075826486572623253 and 
                                     round_tfidf_char_lr_identity_hate <= 0.0033741926308721304 and 
                                     round_tfidf_char_nblr_toxic <= 0.0092530772089958191 and 
                                     round_num_words_title <= 167.5),
          -0.026030854466986941276 * (round_tfidf_word_nostop_nblr_severe_toxic <= 0.0012911264784634113 and 
                                     round_tfidf_word_nostop_nblr_identity_hate <= 0.39716294407844543 and 
                                     round_num_lowercase <= 39.5 and 
                                     round_capital_per_char <= 0.093811109662055969),
          -0.091216193146370722067 * (round_tfidf_word_stop_lr_identity_hate <= 0.0034903497435152531 and 
                                     round_tfidf_word_stop_nblr_toxic <= 0.0050879335030913353 and 
                                     round_lowercase_per_char <= 0.80375593900680542),
          -0.030780650845667920784 * (round_tfidf_word_stop_lr_identity_hate > 0.0034903497435152531 and 
                                     round_tfidf_word_stop_nblr_toxic <= 0.033744275569915771 and 
                                     round_tfidf_word_stop_nblr_identity_hate <= 0.0008749016560614109 and 
                                     round_lowercase_per_char <= 0.80375593900680542),
         -0.0091848221304893246791 * (round_tfidf_word_nostop_nblr_toxic <= 0.052627541124820709 and 
                                     round_tfidf_word_nostop_nblr_identity_hate <= 0.00061076902784407139 and 
                                     round_punctuation_per_char <= 0.10456414520740509 and 
                                     round_num_words_title <= 3.5),
          -0.065002827090112819119 * (round_num_chars <= 72.5 and 
                                     round_punctuation_per_char > 0.10456414520740509),
           0.018276533547408501529 * (round_tfidf_char_nblr_severe_toxic <= 0.0010442386846989393 and 
                                     round_tfidf_union_lr_toxic > 0.062762469053268433 and 
                                     round_tfidf_union_lr_threat > 0.00071564136305823922 and 
                                     round_punctuation_per_char <= 0.092151924967765808),
          0.0093847551590593245724 * (round_tfidf_word_nostop_nblr_toxic > 0.0045890393666923046 and 
                                     round_tfidf_union_lr_threat <= 0.00071564136305823922),
           -0.36734573254222596672 * (round_tfidf_word_stop_nblr_insult <= 0.01430019736289978 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0016753285890445113 and 
                                     round_tfidf_char_nblr_threat <= 0.32041573524475098 and 
                                     round_tfidf_union_nblr_obscene <= 0.12660413980484009),
           -0.18829497213072365902 * (round_tfidf_word_stop_nblr_insult <= 0.01430019736289978 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0016753285890445113 and 
                                     round_tfidf_char_nblr_threat <= 0.32041573524475098 and 
                                     round_tfidf_union_nblr_obscene > 0.12660413980484009),
           -0.24754190613584489689 * (round_tfidf_word_stop_nblr_insult > 0.01430019736289978 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0016753285890445113 and 
                                     round_tfidf_char_nblr_threat <= 0.32041573524475098 and 
                                     round_num_words_lower <= 72.5),
           -0.25349756026147807209 * (round_tfidf_word_stop_nblr_insult > 0.01430019736289978 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0016753285890445113 and 
                                     round_tfidf_char_nblr_threat <= 0.32041573524475098 and 
                                     round_num_words_lower > 72.5),
          -0.061943334835888770595 * (round_tfidf_word_nostop_lr_threat <= 0.0016753285890445113 and 
                                     round_tfidf_char_lr_threat > 0.0010594454361125827 and 
                                     round_tfidf_char_nblr_threat <= 0.32041573524475098 and 
                                     round_tfidf_union_nblr_toxic > 0.0020509739406406879),
          -0.033099392060435009477 * (round_tfidf_word_stop_nblr_toxic <= 0.25208017230033875 and 
                                     round_tfidf_union_nblr_toxic <= 0.70180797576904297 and 
                                     round_num_chars <= 52.5 and 
                                     round_punctuation_per_char > 0.0060882940888404846),
          -0.017529780772022773849 * (round_tfidf_word_nostop_lr_threat > 0.0021260497160255909 and 
                                     round_tfidf_char_lr_threat <= 0.012682989239692688 and 
                                     round_tfidf_union_nblr_severe_toxic <= 0.094145894050598145 and 
                                     round_num_words_title <= 198.5),
          -0.049273810700806254526 * (round_tfidf_word_nostop_nblr_threat <= 0.0016027351375669241 and 
                                     round_tfidf_char_nblr_obscene <= 0.068102419376373291 and 
                                     round_num_words <= 304.5 and 
                                     round_num_unique_words <= 167.5),
          -0.018743416595330193719 * (round_capital_per_char <= 0.10469898581504822 and 
                                     round_lowercase_per_char <= 0.79097294807434082 and 
                                     round_punctuation_per_char <= 0.17320513725280762 and 
                                     round_num_words_lower <= 7.5),
           -0.04866083830688887335 * (round_tfidf_word_stop_lr_identity_hate <= 0.0043405229225754738 and 
                                     round_tfidf_word_nostop_nblr_toxic <= 0.0053379973396658897 and 
                                     round_tfidf_word_nostop_nblr_obscene <= 0.55878949165344238),
           0.042728942824957323599 * (round_tfidf_word_nostop_lr_threat <= 0.002450339961796999 and 
                                     round_tfidf_char_lr_identity_hate > 0.0027712490409612656 and 
                                     round_tfidf_char_nblr_threat <= 0.32041573524475098 and 
                                     round_tfidf_union_lr_insult > 0.015050999820232391),
           0.019646108259929559448 * (round_num_words > 9.5 and 
                                     round_capital_per_char > 0.076489165425300598 and 
                                     round_lowercase_per_char <= 0.80375593900680542),
         -0.0095495058628600471007 * (round_tfidf_word_stop_nblr_identity_hate > 0.00096353731350973248 and 
                                     round_num_words <= 9.5 and 
                                     round_lowercase_per_char <= 0.76631206274032593),
          -0.018069263652572414847 * (round_tfidf_char_nblr_obscene <= 0.0043122284114360809 and 
                                     round_tfidf_union_lr_threat <= 0.0017542364075779915),
           0.042552333495778034422 * (round_tfidf_word_stop_nblr_toxic > 0.41290715336799622 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic <= 0.033181443810462952 and 
                                     round_tfidf_char_lr_threat > 0.0012129424139857292 and 
                                     round_tfidf_char_nblr_insult <= 0.020819973200559616),
          0.0065469635662707510379 * (round_tfidf_word_nostop_nblr_severe_toxic <= 0.033181443810462952 and 
                                     round_tfidf_char_lr_threat > 0.0012129424139857292 and 
                                     round_tfidf_char_nblr_insult > 0.020819973200559616 and 
                                     round_num_stopwords <= 26.5),
           0.040419784424902953368 * (round_tfidf_word_stop_nblr_toxic <= 0.079714886844158173 and 
                                     round_tfidf_word_nostop_nblr_severe_toxic <= 0.033181443810462952 and 
                                     round_tfidf_char_lr_threat <= 0.0012129424139857292 and 
                                     round_tfidf_char_nblr_toxic > 0.0055855484679341316),
           -0.12133084109082722946 * (round_tfidf_word_nostop_lr_threat <= 0.0016753285890445113 and 
                                     round_tfidf_word_nostop_nblr_toxic <= 0.0053532933816313744),
          -0.031869851339751328889 * (round_tfidf_word_nostop_lr_threat <= 0.0016753285890445113 and 
                                     round_tfidf_word_nostop_nblr_toxic > 0.0053532933816313744 and 
                                     round_num_unique_words <= 319.0),
          -0.019893878088906254142 * (round_tfidf_word_stop_nblr_toxic <= 0.43105828762054443 and 
                                     round_tfidf_char_lr_identity_hate > 0.0027712490409612656 and 
                                     round_tfidf_union_lr_insult <= 0.014704696834087372 and 
                                     round_num_lowercase <= 167.5),
          -0.043363311045521366061 * (round_tfidf_word_nostop_nblr_toxic <= 0.049875997006893158 and 
                                     round_tfidf_word_nostop_nblr_identity_hate <= 0.0007512684678658843 and 
                                     round_num_unique_words <= 157.5 and 
                                     round_num_capital <= 32.5),
          0.0025184208702412648473 * (round_tfidf_word_nostop_lr_threat > 0.0023051905445754528 and 
                                     round_tfidf_word_nostop_nblr_identity_hate > 0.0007512684678658843 and 
                                     round_num_unique_words <= 157.5 and 
                                     round_num_capital <= 32.5),
           0.010407226703813823518 * (round_tfidf_word_stop_nblr_obscene > 0.0079073114320635796 and 
                                     round_tfidf_word_nostop_lr_threat <= 0.0024783317930996418 and 
                                     round_tfidf_char_nblr_threat <= 0.0058148917742073536 and 
                                     round_num_unique_words <= 127.5),
           0.015772146029330851025 * (round_num_unique_words <= 175.5 and 
                                     round_num_chars > 54.5 and 
                                     round_capital_per_char <= 0.076489165425300598 and 
                                     round_punctuation_per_char > 0.0060882940888404846),
            -0.1281405982751518402 * (round_punctuation_per_char > 0.17320513725280762),
          0.0043522192664496592643 * (round_tfidf_word_stop_nblr_threat <= 0.00098945270292460918 and 
                                     round_tfidf_word_stop_nblr_insult > 0.0084241870790719986 and 
                                     round_tfidf_char_lr_threat > 0.0012129424139857292 and 
                                     round_num_words_title <= 5.5),
          -0.073460241844265514177 * (round_tfidf_word_stop_lr_identity_hate > 0.0034903497435152531 and 
                                     round_tfidf_word_nostop_nblr_identity_hate <= 0.00096872635185718536 and 
                                     round_tfidf_union_lr_toxic <= 0.06277746707201004 and 
                                     round_num_chars <= 378.5),
           0.022755021877521999785 * (round_tfidf_word_stop_lr_identity_hate > 0.0034903497435152531 and 
                                     round_tfidf_union_lr_toxic > 0.06277746707201004 and 
                                     round_tfidf_union_nblr_threat <= 9.6989620942622423e-05 and 
                                     round_num_lowercase <= 441.5),
           0.012034194832315280727 * (round_tfidf_word_stop_lr_identity_hate > 0.0034903497435152531 and 
                                     round_tfidf_union_lr_toxic > 0.06277746707201004 and 
                                     round_tfidf_union_nblr_threat > 9.6989620942622423e-05 and 
                                     round_num_lowercase <= 441.5),
          0.0065673312617616826786 * (round_tfidf_char_lr_identity_hate > 0.003367643803358078 and 
                                     round_tfidf_union_nblr_severe_toxic > 0.0020710423123091459 and 
                                     round_lowercase_per_char <= 0.80375593900680542),
         -0.0069606206502260551186 * (round_tfidf_word_stop_nblr_threat <= 0.35612630844116211 and 
                                     round_tfidf_char_lr_identity_hate > 0.003367643803358078 and 
                                     round_tfidf_union_nblr_severe_toxic <= 0.0020710423123091459 and 
                                     round_lowercase_per_char <= 0.80375593900680542),
           -0.12648866932522487283 * (round_tfidf_char_lr_identity_hate <= 0.003367643803358078 and 
                                     round_tfidf_char_nblr_toxic <= 0.0093416571617126465 and 
                                     round_lowercase_per_char <= 0.80375593900680542 and 
                                     round_num_words_upper <= 49.5),
           0.041785369523701966499 * (round_tfidf_word_stop_lr_toxic <= 0.10539595037698746 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0016753285890445113 and 
                                     round_tfidf_char_lr_identity_hate > 0.019278885796666145),
           -0.18094309503521782689 * (round_tfidf_word_nostop_lr_threat <= 0.0016753285890445113 and 
                                     round_tfidf_union_nblr_insult <= 0.0035879237111657858),
          -0.097628173922776062543 * (round_tfidf_word_nostop_lr_threat <= 0.0016753285890445113 and 
                                     round_tfidf_union_nblr_insult > 0.0035879237111657858),
            0.01204323731194860414 * (round_tfidf_char_nblr_threat <= 0.32041573524475098 and 
                                     round_num_unique_words <= 233.5 and 
                                     round_punctuation_per_char <= 0.084042668342590332 and 
                                     round_num_words_title <= 198.5),
          -0.017703252139776696344 * (round_tfidf_word_nostop_lr_toxic <= 0.21216484904289246 and 
                                     round_tfidf_word_nostop_nblr_insult <= 0.039685450494289398 and 
                                     round_tfidf_union_lr_obscene <= 0.062956176698207855 and 
                                     round_capital_per_char <= 0.084052205085754395),
          0.0062908550138817782502 * (round_num_chars > 71.5 and 
                                     round_capital_per_char > 0.084052205085754395),
          -0.036148957485701141423 * (round_tfidf_word_stop_nblr_identity_hate <= 0.53769862651824951 and 
                                     round_tfidf_char_lr_toxic <= 0.55876445770263672 and 
                                     round_tfidf_char_lr_identity_hate > 0.003367643803358078 and 
                                     round_capital_per_char <= 0.78736585378646851),
           0.025602261104925763679 * (round_tfidf_word_stop_lr_insult > 0.030131623148918152 and 
                                     round_tfidf_char_lr_toxic > 0.55876445770263672 and 
                                     round_tfidf_char_lr_identity_hate > 0.003367643803358078 and 
                                     round_capital_per_char <= 0.78736585378646851),
           0.023065584442024861245 * (round_tfidf_word_stop_lr_toxic > 0.028721436858177185 and 
                                     round_tfidf_union_nblr_severe_toxic <= 0.050816874951124191 and 
                                     round_punctuation_per_char <= 0.17362318933010101 and 
                                     round_num_words_title <= 0.5),
           0.021815751180339949117 * (round_tfidf_char_nblr_obscene > 0.0089942347258329391 and 
                                     round_tfidf_union_lr_threat <= 0.0017542364075779915 and 
                                     round_num_unique_words <= 128.5),
          -0.052776642730507951351 * (round_tfidf_char_nblr_obscene <= 0.0089942347258329391 and 
                                     round_tfidf_union_lr_toxic <= 0.064603149890899658 and 
                                     round_tfidf_union_lr_threat <= 0.0017542364075779915 and 
                                     round_num_capital <= 17.5),
         -0.0095973237754238705649 * (round_tfidf_word_nostop_nblr_threat <= 0.0015180215705186129 and 
                                     round_tfidf_char_lr_toxic <= 0.55928832292556763 and 
                                     round_tfidf_union_lr_threat > 0.0017542364075779915),
         0.00019190793456530728088 * (round_tfidf_word_stop_nblr_identity_hate > 0.0008749016560614109 and 
                                     round_tfidf_char_lr_identity_hate > 0.0027712490409612656 and 
                                     round_num_words_lower <= 62.5),
          -0.021537316478782334017 * (round_tfidf_word_stop_nblr_identity_hate <= 0.0008749016560614109 and 
                                     round_tfidf_char_lr_identity_hate > 0.0027712490409612656 and 
                                     round_tfidf_char_nblr_insult <= 0.037930697202682495 and 
                                     round_num_capital <= 14.5),
          -0.044115809971555015878 * (round_tfidf_word_stop_nblr_threat <= 0.00098945270292460918 and 
                                     round_tfidf_word_nostop_lr_threat <= 0.0024732840247452259 and 
                                     round_tfidf_union_nblr_obscene <= 0.0032931920140981674 and 
                                     round_num_chars <= 4965.0),
          -0.095797581946717932921 * (round_tfidf_word_stop_nblr_threat <= 0.00098945270292460918 and 
                                     round_tfidf_word_nostop_lr_threat <= 0.0024732840247452259 and 
                                     round_tfidf_union_nblr_obscene > 0.0032931920140981674 and 
                                     round_num_chars <= 4965.0),
          -0.056391999169552189941 * (round_tfidf_word_stop_nblr_threat <= 0.00098945270292460918 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0024732840247452259 and 
                                     round_tfidf_word_nostop_nblr_identity_hate <= 0.0048898826353251934 and 
                                     round_tfidf_union_nblr_obscene <= 0.057185407727956772),
           0.032092932677038082945 * (round_num_capital > 15.5 and 
                                     round_capital_per_char > 0.084052205085754395),
           0.077896142696466497135 * (round_tfidf_word_nostop_lr_toxic <= 0.21216484904289246 and 
                                     round_tfidf_word_nostop_nblr_toxic <= 0.30763897299766541 and 
                                     round_tfidf_char_nblr_obscene > 0.14230549335479736 and 
                                     round_capital_per_char <= 0.084052205085754395),
           0.061002772342906082248 * (round_tfidf_word_nostop_lr_toxic <= 0.21216484904289246 and 
                                     round_tfidf_word_nostop_nblr_toxic > 0.30763897299766541 and 
                                     round_num_words <= 29.5 and 
                                     round_capital_per_char <= 0.084052205085754395),
         -0.0063314091181964852334 * (round_tfidf_char_lr_identity_hate > 0.0033741926308721304 and 
                                     round_tfidf_union_nblr_threat <= 0.020551387220621109 and 
                                     round_tfidf_union_nblr_insult <= 0.35093653202056885 and 
                                     round_lowercase_per_char <= 0.79097294807434082),
           0.016725497547865751657 * (round_tfidf_word_nostop_lr_identity_hate <= 0.11752128601074219 and 
                                     round_tfidf_char_lr_identity_hate > 0.0033741926308721304 and 
                                     round_tfidf_union_nblr_insult > 0.35093653202056885 and 
                                     round_lowercase_per_char <= 0.79097294807434082),
           0.021412099294829203128 * (round_tfidf_char_nblr_severe_toxic > 0.0010261342395097017 and 
                                     round_num_capital <= 22.5 and 
                                     round_num_words_title <= 26.5),
         -0.0087721307038551524921 * (round_tfidf_char_nblr_identity_hate <= 0.0023267064243555069 and 
                                     round_punctuation_per_char <= 0.092125140130519867 and 
                                     round_num_words_title <= 3.5),
           0.018034168491373347987 * (round_tfidf_word_nostop_lr_toxic <= 0.19168548285961151 and 
                                     round_tfidf_word_nostop_nblr_obscene <= 0.55873286724090576 and 
                                     round_num_words > 10.5 and 
                                     round_lowercase_per_char <= 0.80375593900680542),
           0.015386271890266868881 * (round_tfidf_word_stop_lr_identity_hate > 0.0032491437159478664 and 
                                     round_tfidf_word_nostop_lr_toxic <= 0.19169607758522034 and 
                                     round_tfidf_char_nblr_obscene <= 0.14726679027080536 and 
                                     round_tfidf_union_lr_insult > 0.014704696834087372),
          0.0077316269622795183447 * (round_tfidf_word_stop_lr_identity_hate > 0.0032491437159478664 and 
                                     round_tfidf_word_nostop_lr_toxic <= 0.19169607758522034 and 
                                     round_tfidf_char_nblr_obscene > 0.14726679027080536 and 
                                     round_tfidf_union_lr_insult > 0.014704696834087372),
             -0.032726513503609167 * (round_tfidf_word_nostop_lr_threat > 0.0024780239909887314 and 
                                     round_tfidf_word_nostop_nblr_obscene <= 0.86735939979553223 and 
                                     round_tfidf_word_nostop_nblr_threat <= 0.00059638684615492821 and 
                                     round_tfidf_char_lr_threat <= 0.014524084515869617),
          -0.021199388020144687567 * (round_tfidf_char_nblr_obscene <= 0.012221885845065117 and 
                                     round_num_unique_words <= 153.5 and 
                                     round_num_capital <= 22.5 and 
                                     round_lowercase_per_char <= 0.76138615608215332),
          0.0049010600684447104547 * (round_tfidf_char_nblr_obscene > 0.012221885845065117 and 
                                     round_num_unique_words <= 153.5 and 
                                     round_num_capital <= 22.5 and 
                                     round_lowercase_per_char <= 0.76138615608215332),
           0.049235818538846148895 * (round_num_unique_words <= 153.5 and 
                                     round_num_capital > 22.5 and 
                                     round_num_words_title <= 14.5),
           0.045194451069736171767 * (round_num_unique_words <= 35.5 and 
                                     round_num_capital <= 22.5 and 
                                     round_lowercase_per_char > 0.76138615608215332),
          -0.084264846675083160399 * (round_tfidf_word_nostop_nblr_toxic > 0.029401363804936409 and 
                                     round_num_unique_words > 153.5),
           0.020153925808441663925 * (round_tfidf_union_nblr_toxic > 0.12190251797437668 and 
                                     round_capital_per_char > 0.029149208217859268 and 
                                     round_punctuation_per_char <= 0.084042668342590332 and 
                                     round_num_words_title > 3.5),
           0.013859944081606279803 * (round_tfidf_word_stop_nblr_insult > 0.028879648074507713 and 
                                     round_tfidf_union_nblr_toxic <= 0.12190251797437668 and 
                                     round_punctuation_per_char <= 0.084042668342590332 and 
                                     round_num_words_title > 3.5),
           0.066312786425068714902 * (round_tfidf_word_stop_nblr_identity_hate <= 0.00073811097536236048 and 
                                     round_punctuation_per_char <= 0.084042668342590332 and 
                                     round_num_words_title <= 3.5),
           0.079092409791734277769 * (round_tfidf_word_stop_nblr_identity_hate > 0.00073811097536236048 and 
                                     round_punctuation_per_char <= 0.084042668342590332 and 
                                     round_num_words_title <= 3.5),
           0.023814654650446825945 * (round_tfidf_word_stop_lr_insult > 0.013748333789408207 and 
                                     round_tfidf_word_nostop_lr_threat <= 0.0024495783727616072 and 
                                     round_tfidf_word_nostop_nblr_insult > 0.0039151487872004509 and 
                                     round_num_unique_words <= 128.5),
           0.022017713040246807144 * (round_tfidf_word_stop_lr_insult > 0.013748333789408207 and 
                                     round_tfidf_word_nostop_lr_threat > 0.0024495783727616072 and 
                                     round_tfidf_word_nostop_nblr_insult > 0.0039151487872004509 and 
                                     round_num_unique_words <= 128.5),
          -0.020571379266837306471 * (round_tfidf_word_stop_lr_severe_toxic > 0.0032298632431775331 and 
                                     round_tfidf_word_stop_nblr_insult <= 0.011443238705396652 and 
                                     round_tfidf_word_nostop_nblr_threat <= 0.1441284716129303 and 
                                     round_tfidf_char_nblr_severe_toxic <= 0.0010261342395097017),
            0.01627806557508034227 * (round_tfidf_word_stop_lr_severe_toxic <= 0.0032298632431775331 and 
                                     round_tfidf_word_stop_nblr_toxic <= 0.077430978417396545 and 
                                     round_tfidf_word_nostop_nblr_threat <= 0.1441284716129303 and 
                                     round_tfidf_char_nblr_severe_toxic <= 0.0010261342395097017),
         0.00032435921043699050825 * (round_tfidf_char_nblr_severe_toxic > 0.0010261342395097017 and 
                                     round_tfidf_union_nblr_threat <= 0.00021988409571349621),
           -0.02151936348962620732 * (round_tfidf_word_stop_nblr_identity_hate <= 0.034820683300495148 and 
                                     round_tfidf_word_nostop_nblr_obscene <= 0.73903048038482666 and 
                                     round_tfidf_char_nblr_threat <= 0.28832921385765076 and 
                                     round_num_chars <= 4957.5),
           0.053421314792334799459 * (round_tfidf_word_stop_nblr_identity_hate > 0.034820683300495148 and 
                                     round_tfidf_word_nostop_nblr_obscene <= 0.73903048038482666 and 
                                     round_tfidf_union_nblr_obscene <= 0.01797395758330822),
          -0.021175471804599388381 * (round_tfidf_word_stop_nblr_threat <= 0.00082137773279100657 and 
                                     round_tfidf_char_lr_threat > 0.0030742133967578411 and 
                                     round_tfidf_char_lr_insult > 0.072295449674129486),
           0.016274431430487741601 * (round_tfidf_word_stop_nblr_threat <= 0.00082137773279100657 and 
                                     round_tfidf_char_lr_threat <= 0.0030742133967578411 and 
                                     round_tfidf_union_nblr_obscene > 0.0048848660662770271 and 
                                     round_num_unique_words <= 128.5),
         -0.0014453328392037122913 * (round_tfidf_word_stop_nblr_threat <= 0.00082137773279100657 and 
                                     round_tfidf_char_lr_threat <= 0.0030742133967578411 and 
                                     round_tfidf_union_nblr_obscene <= 0.0048848660662770271 and 
                                     round_num_words_upper <= 46.5),
          -0.031102653633528178773 * (round_tfidf_word_stop_nblr_threat <= 0.00082137773279100657 and 
                                     round_tfidf_char_lr_threat > 0.0030742133967578411 and 
                                     round_tfidf_char_lr_insult <= 0.072295449674129486 and 
                                     round_capital_per_char <= 0.82823538780212402),
         -0.0064331493808673558682 * (round_tfidf_union_nblr_severe_toxic <= 0.094145894050598145 and 
                                     round_punctuation_per_char <= 0.23420877754688263 and 
                                     0.5 < round_num_words_title <= 198.5),
           0.027315671739392317907 * (round_tfidf_word_stop_nblr_insult > 0.047478176653385162 and 
                                     round_tfidf_word_nostop_lr_toxic <= 0.195844367146492 and 
                                     round_tfidf_char_nblr_obscene <= 0.03449895977973938),
          0.0047419314285526790315 * (round_tfidf_word_stop_nblr_insult <= 0.047478176653385162 and 
                                     round_tfidf_word_nostop_lr_toxic <= 0.195844367146492 and 
                                     round_tfidf_word_nostop_nblr_identity_hate > 0.0034496004227548838 and 
                                     round_tfidf_char_nblr_obscene <= 0.03449895977973938),
           0.026169010082161454217 * (round_tfidf_word_stop_nblr_identity_hate > 0.0075826486572623253 and 
                                     round_tfidf_union_nblr_toxic <= 0.13640564680099487),
            0.03022203490090093897 * (round_tfidf_word_stop_nblr_identity_hate <= 0.0075826486572623253 and 
                                     round_tfidf_char_lr_identity_hate <= 0.0042499522678554058 and 
                                     round_tfidf_union_nblr_obscene > 0.0038297770079225302),
           0.026164701888065609175 * (round_tfidf_word_stop_nblr_obscene > 0.033205531537532806 and 
                                     round_tfidf_word_nostop_lr_toxic <= 0.21216484904289246 and 
                                     round_capital_per_char <= 0.084052205085754395),
           0.032877517974643832854 * (round_tfidf_word_stop_nblr_obscene <= 0.033205531537532806 and 
                                     round_tfidf_word_nostop_lr_toxic <= 0.21216484904289246 and 
                                     round_tfidf_char_nblr_insult > 0.0853375643491745 and 
                                     round_capital_per_char <= 0.084052205085754395),
           0.024692345719710240282 * (round_capital_per_char > 0.084052205085754395 and 
                                     round_num_words_upper > 0.5),
          -0.053223198128682210062 * (round_tfidf_word_stop_nblr_threat <= 0.00082137773279100657 and 
                                     round_tfidf_char_nblr_threat <= 0.00027362236869521439 and 
                                     round_tfidf_char_nblr_insult <= 0.02411966398358345 and 
                                     round_tfidf_union_lr_obscene <= 0.015992503613233566),
           0.015525798367427134233 * (round_tfidf_word_stop_nblr_threat <= 0.00082137773279100657 and 
                                     round_tfidf_char_nblr_threat <= 0.00027362236869521439 and 
                                     round_tfidf_char_nblr_insult > 0.02411966398358345 and 
                                     round_tfidf_union_lr_obscene <= 0.015992503613233566),
           0.040372659226526969034 * (round_tfidf_word_nostop_nblr_toxic > 0.16867250204086304 and 
                                     round_punctuation_per_char > 0.084042668342590332),
          -0.017285853807640281371 * (round_tfidf_word_stop_lr_identity_hate > 0.0031208866275846958 and 
                                     round_lowercase_per_char > 0.54085850715637207 and 
                                     round_punctuation_per_char <= 0.084042668342590332),
           0.033632020887843665224 * (round_tfidf_union_nblr_obscene > 0.0093359760940074921 and 
                                     round_num_chars > 52.5 and 
                                     round_capital_per_char > 0.049813084304332733 and 
                                     round_lowercase_per_char <= 0.7856488823890686),
          -0.027731743035420874238 * (round_tfidf_word_nostop_nblr_identity_hate <= 0.0012946997303515673 and 
                                     round_num_chars <= 52.5 and 
                                     round_capital_per_char <= 0.72760999202728271 and 
                                     round_lowercase_per_char <= 0.7856488823890686),
         -0.0040916921402263072005 * (round_tfidf_union_nblr_toxic <= 0.12352132797241211 and 
                                     round_num_chars > 52.5 and 
                                     round_capital_per_char <= 0.049813084304332733 and 
                                     round_lowercase_per_char <= 0.7856488823890686)    ])

def get_type_conversion():
    return {}
INDICATOR_COLS = []

IMPUTE_VALUES = {
    u'tfidf_union_nblr_toxic': 0.005778,
    u'lowercase_per_char': 0.739680,
    u'tfidf_word_nostop_nblr_threat': 0.000117,
    u'tfidf_char_lr_identity_hate': 0.003162,
    u'tfidf_word_stop_lr_toxic': 0.026923,
    u'num_stopwords': 16.000000,
    u'tfidf_word_nostop_lr_toxic': 0.028748,
    u'tfidf_word_nostop_nblr_toxic': 0.007404,
    u'tfidf_char_lr_threat': 0.001475,
    u'tfidf_char_lr_obscene': 0.009987,
    u'tfidf_char_lr_insult': 0.009273,
    u'tfidf_word_nostop_lr_severe_toxic': 0.003464,
    u'num_punctuations': 8.000000,
    u'tfidf_word_nostop_nblr_obscene': 0.002376,
    u'num_words': 36.000000,
    u'tfidf_word_nostop_lr_threat': 0.001856,
    u'tfidf_word_stop_nblr_identity_hate': 0.000419,
    u'tfidf_union_lr_obscene': 0.005773,
    u'tfidf_union_lr_insult': 0.005618,
    u'num_lowercase': 148.000000,
    u'tfidf_word_stop_lr_identity_hate': 0.004229,
    u'tfidf_char_nblr_toxic': 0.008745,
    u'tfidf_word_stop_lr_obscene': 0.013559,
    u'tfidf_word_stop_lr_threat': 0.001826,
    u'tfidf_word_stop_lr_severe_toxic': 0.003175,
    u'tfidf_union_nblr_severe_toxic': 0.000312,
    u'tfidf_word_stop_nblr_obscene': 0.002627,
    u'tfidf_union_nblr_threat': 0.000062,
    u'tfidf_word_stop_nblr_insult': 0.002816,
    u'tfidf_word_nostop_nblr_insult': 0.002860,
    u'tfidf_union_lr_threat': 0.000752,
    u'tfidf_word_nostop_lr_insult': 0.014115,
    u'tfidf_char_nblr_threat': 0.000132,
    u'tfidf_word_nostop_nblr_severe_toxic': 0.000347,
    u'tfidf_char_nblr_severe_toxic': 0.000419,
    u'tfidf_char_nblr_insult': 0.003601,
    u'punctuation_per_char': 0.038095,
    u'tfidf_union_nblr_obscene': 0.002504,
    u'tfidf_char_lr_toxic': 0.020086,
    u'num_words_upper': 1.000000,
    u'tfidf_union_lr_toxic': 0.011900,
    u'num_words_lower': 28.000000,
    u'tfidf_char_nblr_identity_hate': 0.000421,
    u'tfidf_union_lr_severe_toxic': 0.001324,
    u'tfidf_word_stop_nblr_severe_toxic': 0.000363,
    u'tfidf_union_nblr_insult': 0.002802,
    u'tfidf_word_stop_nblr_toxic': 0.007456,
    u'num_chars': 205.000000,
    u'tfidf_word_stop_nblr_threat': 0.000130,
    u'num_capital': 7.000000,
    u'tfidf_word_nostop_lr_obscene': 0.013687,
    u'tfidf_word_nostop_lr_identity_hate': 0.004079,
    u'tfidf_char_nblr_obscene': 0.003290,
    u'tfidf_word_stop_lr_insult': 0.012813,
    u'num_unique_words': 31.000000,
    u'num_words_title': 5.000000,
    u'capital_per_char': 0.031757,
    u'tfidf_char_lr_severe_toxic': 0.002290,
    u'tfidf_word_nostop_nblr_identity_hate': 0.000360,}


def bag_of_words(text):
    """ set of whole words  in a block of text """
    if type(text) == float:
        return set()

    return set(word.lower() for word in
               re.findall(ur'\w+', text, re.UNICODE | re.IGNORECASE))


def parse_date(x, date_format):
    """ convert date strings to numeric values. """
    try:
        if isinstance(x, np.float64):
            x = long(x)
        if '%M' in date_format:
            temp = str(x)
            if re.search('[\+-][0-9]+$', temp):
                temp = re.sub('[\+-][0-9]+$', '', temp)

            return calendar.timegm(datetime.strptime(temp, date_format).timetuple())
        else:
            return datetime.strptime(str(x), date_format).toordinal()
    except:
        return float('nan')


def parse_percentage(s):
    """ remove percent sign so percentage variables can be converted to numeric """
    try:
        return float(s.replace('%', ''))
    except:
        return float('nan')

def parse_nonstandard_na(s):
    """ if a column contains numbers and a unique non-numeric,
        then the non-numeric is considered to be N/A
    """
    try:
        ret = float(s)
        if np.isinf(ret):
            return float('nan')
        return ret
    except:
        return float('nan')

def parse_length(s):
    """ convert feet and inches as string to inches as numeric """
    try:
        if '"' in s and "'" in s:
            sp = s.split("'")
            return float(sp[0]) * 12 + float(sp[1].replace('"', ''))
        else:
            if "'" in s:
                return float(s.replace("'", '')) * 12
            else:
                return float(s.replace('"', ''))
    except:
        return float('nan')

def parse_currency(s):
    """ strip currency characters and commas from currency columns """
    if not isinstance(s, unicode):
        return float('nan')
    s = re.sub(u'[\$\u20AC\u00A3\uFFE1\u00A5\uFFE5]|(EUR)', '', s)
    s = s.replace(',', '')
    try:
        return float(s)
    except:
        return float('nan')


def parse_currency_replace_cents_period(val, currency_symbol):
    try:
        if np.isnan(val):
            return val
    except TypeError:
        pass
    if not isinstance(val, basestring):
        raise ValueError('Found wrong value for currency: {}'.format(val))
    try:
        val = val.replace(currency_symbol, "", 1)
        val = val.replace(" ", "")
        val = val.replace(",", "")
        val = float(val)
    except ValueError:
        val = float('nan')
    return val


def parse_currency_replace_cents_comma(val, currency_symbol):
    try:
        if np.isnan(val):
            return val
    except TypeError:
        pass
    if not isinstance(val, basestring):
        raise ValueError('Found wrong value for currency: {}'.format(val))
    try:
        val = val.replace(currency_symbol, "", 1)
        val = val.replace(" ", "")
        val = val.replace(".", "")
        val = val.replace(",", ".")
        val = float(val)
    except ValueError:
        val = float('nan')
    return val


def parse_currency_replace_no_cents(val, currency_symbol):
    try:
        if np.isnan(val):
            return val
    except TypeError:
        pass
    if not isinstance(val, basestring):
        raise ValueError('Found wrong value for currency: {}'.format(val))
    try:
        val = val.replace(currency_symbol, "", 1)
        val = val.replace(" ", "")
        val = val.replace(",", "")
        val = val.replace(".", "")
        val = float(val)
    except ValueError:
        val = float('nan')
    return val

def parse_numeric_types(ds):
    """ convert strings with numeric types (date, currency, etc.)
        to actual numeric values """
    TYPE_CONVERSION = get_type_conversion()
    for col in ds.columns:
        if col in TYPE_CONVERSION:
            convert_func = TYPE_CONVERSION[col]['convert_func']
            convert_args = TYPE_CONVERSION[col]['convert_args']
            ds[col] = ds[col].apply(convert_func, args=convert_args)
    return ds

def sanitize_name(name):
    safe = name.strip().replace("-", "_").replace("$", "_").replace(".", "_")
    safe = safe.replace("{", "_").replace("}", "_")
    safe = safe.replace('"', '_')
    return safe

def rename_columns(ds):
    new_names = {}
    existing_names = set()
    disambiguation = {}
    blank_index = 0
    for old_col in ds.columns:
        col = sanitize_name(old_col)
        if col == '':
            col = 'Unnamed: %d' % blank_index
            blank_index += 1
        if col in existing_names:
            suffix = '_%d' % disambiguation.setdefault(col, 1)
            disambiguation[col] += 1
            col = col + suffix
        existing_names.add(col)
        new_names[old_col] = col
    ds.rename(columns=new_names, inplace=True)
    return ds

def add_missing_indicators(ds):
    for col in INDICATOR_COLS:
        ds[col + '-mi'] = ds[col].isnull().astype(int)
    return ds

def impute_values(ds):
    for col in ds:
        if col in IMPUTE_VALUES:
            ds.loc[ds[col].isnull(), col] = IMPUTE_VALUES[col]
    return ds

BIG_LEVELS = {
}


SMALL_NULLS = {
}


VAR_TYPES = {
    u'tfidf_union_nblr_toxic': 'N',
    u'lowercase_per_char': 'N',
    u'tfidf_word_nostop_nblr_threat': 'N',
    u'tfidf_char_lr_identity_hate': 'N',
    u'tfidf_word_stop_lr_toxic': 'N',
    u'num_capital': 'N',
    u'tfidf_word_nostop_lr_toxic': 'N',
    u'tfidf_char_nblr_insult': 'N',
    u'tfidf_word_nostop_nblr_toxic': 'N',
    u'tfidf_char_lr_obscene': 'N',
    u'tfidf_char_lr_insult': 'N',
    u'tfidf_word_nostop_lr_severe_toxic': 'N',
    u'num_punctuations': 'N',
    u'tfidf_word_nostop_nblr_obscene': 'N',
    u'num_words': 'N',
    u'tfidf_word_nostop_lr_threat': 'N',
    u'tfidf_word_stop_nblr_identity_hate': 'N',
    u'tfidf_union_lr_obscene': 'N',
    u'tfidf_union_lr_insult': 'N',
    u'num_lowercase': 'N',
    u'tfidf_word_stop_lr_identity_hate': 'N',
    u'tfidf_char_nblr_toxic': 'N',
    u'tfidf_word_stop_lr_obscene': 'N',
    u'tfidf_word_stop_lr_threat': 'N',
    u'tfidf_word_stop_lr_severe_toxic': 'N',
    u'tfidf_union_nblr_severe_toxic': 'N',
    u'tfidf_word_stop_nblr_obscene': 'N',
    u'tfidf_union_nblr_threat': 'N',
    u'tfidf_word_stop_nblr_insult': 'N',
    u'tfidf_word_nostop_nblr_insult': 'N',
    u'tfidf_union_lr_threat': 'N',
    u'comment_text': 'T',
    u'tfidf_word_nostop_lr_insult': 'N',
    u'tfidf_word_nostop_nblr_severe_toxic': 'N',
    u'tfidf_char_nblr_severe_toxic': 'N',
    u'tfidf_word_stop_nblr_threat': 'N',
    u'tfidf_word_stop_nblr_toxic': 'N',
    u'punctuation_per_char': 'N',
    u'tfidf_union_nblr_obscene': 'N',
    u'tfidf_char_lr_toxic': 'N',
    u'num_words_upper': 'N',
    u'tfidf_union_lr_toxic': 'N',
    u'num_words_lower': 'N',
    u'tfidf_char_lr_threat': 'N',
    u'num_words_title': 'N',
    u'tfidf_word_stop_nblr_severe_toxic': 'N',
    u'tfidf_union_nblr_insult': 'N',
    u'tfidf_char_nblr_identity_hate': 'N',
    u'capital_per_char': 'N',
    u'num_chars': 'N',
    u'tfidf_char_lr_severe_toxic': 'N',
    u'tfidf_word_nostop_lr_obscene': 'N',
    u'tfidf_word_nostop_lr_identity_hate': 'N',
    u'tfidf_char_nblr_obscene': 'N',
    u'tfidf_word_stop_lr_insult': 'N',
    u'num_unique_words': 'N',
    u'tfidf_union_lr_severe_toxic': 'N',
    u'num_stopwords': 'N',
    u'tfidf_char_nblr_threat': 'N',
    u'tfidf_word_nostop_nblr_identity_hate': 'N',
}


def combine_small_levels(ds):
    for col in ds:
        if BIG_LEVELS.get(col, None) is not None:
            mask = np.logical_and(~ds[col].isin(BIG_LEVELS[col]), ds[col].notnull())
            if np.any(mask):
                ds.loc[mask, col] = 'small_count'
        if SMALL_NULLS.get(col):
            mask = ds[col].isnull()
            if np.any(mask):
                ds.loc[mask, col] = 'small_count'
        if VAR_TYPES.get(col) == 'C' or VAR_TYPES.get(col) == 'T':
            mask = ds[col].isnull()
            if np.any(mask):
                if ds[col].dtype == float:
                    ds[col] = ds[col].astype(object)
                ds.loc[mask, col] = 'nan'
    return ds

# N/A strings in addition to the ones used by Pandas read_csv()
NA_VALUES = ['null', 'na', 'n/a', '#N/A', 'N/A', '?', '.', '', 'Inf', 'INF', 'inf', '-inf', '-Inf', '-INF', ' ', 'None', 'NaN', '-nan', 'NULL', 'NA', '-1.#IND', '1.#IND', '-1.#QNAN', '1.#QNAN', '#NA', '#N/A N/A', '-NaN', 'nan']

# True/False strings in addition to the ones used by Pandas read_csv()
TRUE_VALUES = ['TRUE', 'True', 'true']
FALSE_VALUES = ['FALSE', 'False', 'false']

DEFAULT_ENCODING = 'utf8'

REQUIRED_COLUMNS = [u"tfidf_union_nblr_toxic",u"lowercase_per_char",u"num_stopwords",u"tfidf_char_lr_identity_hate",u"tfidf_word_stop_lr_toxic",u"tfidf_word_nostop_lr_toxic",u"tfidf_word_nostop_nblr_toxic",u"tfidf_char_lr_obscene",u"tfidf_char_lr_insult",u"tfidf_word_nostop_lr_severe_toxic",u"tfidf_char_nblr_toxic",u"tfidf_word_nostop_nblr_obscene",u"num_words",u"tfidf_char_nblr_identity_hate",u"tfidf_word_nostop_nblr_severe_toxic",u"tfidf_word_stop_nblr_identity_hate",u"num_words_upper",u"tfidf_union_lr_insult",u"num_lowercase",u"tfidf_word_stop_lr_identity_hate",u"tfidf_char_lr_threat",u"tfidf_word_stop_lr_obscene",u"tfidf_union_nblr_severe_toxic",u"tfidf_word_stop_lr_threat",u"tfidf_word_stop_lr_severe_toxic",u"tfidf_word_stop_nblr_obscene",u"tfidf_word_stop_nblr_threat",u"tfidf_union_nblr_threat",u"tfidf_word_stop_nblr_insult",u"tfidf_word_nostop_nblr_insult",u"num_chars",u"comment_text",u"tfidf_word_nostop_lr_insult",u"tfidf_union_lr_threat",u"num_punctuations",u"tfidf_word_nostop_lr_threat",u"tfidf_char_nblr_severe_toxic",u"tfidf_char_nblr_insult",u"punctuation_per_char",u"tfidf_union_nblr_obscene",u"tfidf_char_lr_toxic",u"tfidf_word_nostop_nblr_threat",u"tfidf_union_lr_obscene",u"tfidf_union_lr_toxic",u"num_words_lower",u"num_unique_words",u"tfidf_union_lr_severe_toxic",u"tfidf_word_stop_nblr_severe_toxic",u"tfidf_union_nblr_insult",u"tfidf_word_stop_nblr_toxic",u"num_capital",u"tfidf_word_nostop_lr_obscene",u"tfidf_word_nostop_lr_identity_hate",u"tfidf_char_nblr_obscene",u"tfidf_char_nblr_threat",u"tfidf_word_stop_lr_insult",u"num_words_title",u"capital_per_char",u"tfidf_char_lr_severe_toxic",u"tfidf_word_nostop_nblr_identity_hate"]


def validate_columns(column_list):
    if set(REQUIRED_COLUMNS) <= set(column_list):
        return True
    else :
        raise ValueError("Required columns missing: %s" %
                         (set(REQUIRED_COLUMNS) - set(column_list)))

def convert_bool(ds):
    TYPE_CONVERSION = get_type_conversion()
    for col in ds.columns:
        if VAR_TYPES.get(col) == 'C' and ds[col].dtype in (int, float):
            mask = ds[col].notnull()
            ds[col] = ds[col].astype(object)
            ds.loc[mask, col] = ds.loc[mask, col].astype(unicode)
        elif VAR_TYPES.get(col) == 'N' and ds[col].dtype == bool:
            ds[col] = ds[col].astype(float)
        elif ds[col].dtype == bool:
            ds[col] = ds[col].astype(unicode)
        elif ds[col].dtype == object:
            if VAR_TYPES.get(col) == 'N' and col not in TYPE_CONVERSION:
                mask = ds[col].apply(lambda x: x in TRUE_VALUES)
                if np.any(mask):
                    ds.loc[mask, col] = 1
                mask = ds[col].apply(lambda x: x in FALSE_VALUES)
                if np.any(mask):
                    ds.loc[mask, col] = 0
                ds[col] = ds[col].astype(float)
            elif TYPE_CONVERSION.get(col) is None:
                mask = ds[col].notnull()
                ds.loc[mask, col] = ds.loc[mask, col].astype(unicode)
    return ds

def get_dtypes():
    return {a: object for a, b in VAR_TYPES.items() if b == 'C'}

def predict_dataframe(ds):
    return ds.apply(predict, axis=1)

def run(dataset_path, output_path, encoding=None):
    if encoding is None:
        encoding = DEFAULT_ENCODING

    ds = pd.read_csv(dataset_path, na_values=NA_VALUES, low_memory=False,
                     dtype=get_dtypes(), encoding=encoding)
    ds = rename_columns(ds)
    ds = convert_bool(ds)
    validate_columns(ds.columns)
    ds = parse_numeric_types(ds)
    ds = add_missing_indicators(ds)
    ds = impute_values(ds)
    ds = combine_small_levels(ds)
    prediction = 1/(1 + np.exp(-predict_dataframe(ds)))
    prediction_file = output_path
    prediction.name = 'Prediction'
    prediction.to_csv(prediction_file, header=True, index_label='Index')


def _construct_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Make offline predictions with DataRobot Prime')

    parser.add_argument(
        '--encoding',
        type=str,
        help=('the encoding of the dataset you are going to make predictions with. '
              'DataRobot Prime defaults to UTF-8 if not otherwise specified. See the '
              '"Codecs" column of the Python-supported standards chart '
              '(https://docs.python.org/2/library/codecs.html#standard-encodings) '
              'for possible alternative entries.'),
        metavar='<encoding>'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help=('a .csv file (your dataset); columns must correspond to the '
              'feature set used to generate the DataRobot Prime model.'),
        metavar='<data_file>'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='the filename where DataRobot writes the results.',
        metavar='<output_file>'
    )

    return parser


def _parse_command(args):
    parser = _construct_parser()
    parsed_args = parser.parse_args(args[1:])

    if parsed_args.encoding is None:
        sys.stderr.write('Warning: For input data encodings other than the standard utf-8, '
                         'see documentation at https://app.datarobot.com/docs/users-guide/more-info/tabs/prime-examples.html')
        parsed_args.encoding = DEFAULT_ENCODING

    return parsed_args


if __name__ == '__main__':
    args = _parse_command(sys.argv)
    run(args.input_path, args.output_path, encoding=args.encoding)
