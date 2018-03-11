import re
import string

import numpy as np

from textblob import TextBlob
from afinn import Afinn
from textstat.textstat import textstat

from nltk.corpus import stopwords

from utils import print_step
from preprocess import normalize_text


def has_verb_you(ws):
    if len(ws) <= 1:
        return False
    verbs = [i for i, w in enumerate(ws) if 'VB' in w[1]]
    after_verbs = [ws[v+1][0] if len(ws) > v+1 else '' for v in verbs]
    return any(['you' in av for av in after_verbs])

def has_you_verb(ws):
    if len(ws) <= 1:
        return False
    yous = [i for i, w in enumerate(ws) if w[0] == 'you']
    after_yous = [ws[y+1][1] if len(ws) > y+1 else '' for y in yous]
    has_adjective = any(['JJ' in w[1] for w in ws])
    return any(['VBP' in ay for ay in after_yous]) and has_adjective


def add_features(train, test):
    train2 = train.copy()  # Avoid overwrites
    test2 = test.copy()

    # Basic features
    print_step('BASIC FE 1/30')
    train2['num_words'] = train2['comment_text'].apply(lambda x: len(str(x).split()))
    test2['num_words'] = test2['comment_text'].apply(lambda x: len(str(x).split()))

    print_step('BASIC FE 2/30')
    train2['num_unique_words'] = train2['comment_text'].apply(lambda x: len(set(str(x).lower().split())))
    test2['num_unique_words'] = test2['comment_text'].apply(lambda x: len(set(str(x).lower().split())))

    print_step('BASIC FE 3/30')
    train2['unique_words_per_word'] = train2['num_unique_words'] / (train2['num_words'] + 0.0001)
    test2['unique_words_per_word'] = test2['num_unique_words'] / (test2['num_words'] + 0.0001)

    print_step('BASIC FE 4/30')
    train2['num_chars'] = train2['comment_text'].apply(lambda x: len(str(x)))
    test2['num_chars'] = test2['comment_text'].apply(lambda x: len(str(x)))

    print_step('BASIC FE 5/30')
    train2['num_capital'] = train2['comment_text'].apply(lambda x: len([c for c in x if c.isupper()]))
    test2['num_capital'] = test2['comment_text'].apply(lambda x: len([c for c in x if c.isupper()]))

    print_step('BASIC FE 6/30')
    train2['num_lowercase'] = train2['comment_text'].apply(lambda x: len([c for c in x if c.islower()]))
    test2['num_lowercase'] = test2['comment_text'].apply(lambda x: len([c for c in x if c.islower()]))

    print_step('BASIC FE 7/30')
    train2['capital_per_char'] = train2['num_capital'] / train2['num_chars']
    test2['capital_per_char'] = test2['num_capital'] / test2['num_chars']

    print_step('BASIC FE 8/30')
    train2['lowercase_per_char'] = train2['num_lowercase'] / train2['num_chars']
    test2['lowercase_per_char'] = test2['num_lowercase'] / test2['num_chars']

    print_step('BASIC FE 9/30')
    stop_words = {x: 1 for x in stopwords.words('english')}
    train2['num_stopwords'] = train2['comment_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))
    test2['num_stopwords'] = test2['comment_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))

    print_step('BASIC FE 10/30')
    train2['num_punctuations'] = train2['comment_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    test2['num_punctuations'] = test2['comment_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    print_step('BASIC FE 11/30')
    train2['punctuation_per_char'] = train2['num_punctuations'] / train2['num_chars']
    test2['punctuation_per_char'] = test2['num_punctuations'] / test2['num_chars']

    print_step('BASIC FE 12/30')
    train2['num_words_upper'] = train2['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    test2['num_words_upper'] = test2['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    print_step('BASIC FE 13/30')
    train2['num_words_lower'] = train2['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
    test2['num_words_lower'] = test2['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))

    print_step('BASIC FE 14/30')
    train2['num_words_title'] = train2['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    test2['num_words_title'] = test2['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    print_step('BASIC FE 15/30')
    train2['chars_per_word'] = train2['num_chars'] / train2['num_words']
    test2['chars_per_word'] = test2['num_chars'] / test2['num_words']

    print_step('BASIC FE 16/30')
    train2['sentence'] = train2['comment_text'].apply(lambda x: [s for s in re.split(r'[.!?\n]+', str(x))])
    test2['sentence'] = test2['comment_text'].apply(lambda x: [s for s in re.split(r'[.!?\n]+', str(x))])

    print_step('BASIC FE 17/30')
    train2['num_sentence'] = train2['sentence'].apply(lambda x: len(x))
    test2['num_sentence'] = test2['sentence'].apply(lambda x: len(x))

    print_step('BASIC FE 18/30')
    train2['sentence_mean'] = train2.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: np.mean(x))
    test2['sentence_mean'] = test2.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: np.mean(x))

    print_step('BASIC FE 19/30')
    train2['sentence_max'] = train2.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: max(x) if len(x) > 0 else 0)
    test2['sentence_max'] = test2.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: max(x) if len(x) > 0 else 0)

    print_step('BASIC FE 20/30')
    train2['sentence_min'] = train2.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: min(x) if len(x) > 0 else 0)
    test2['sentence_min'] = test2.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: min(x) if len(x) > 0 else 0)

    print_step('BASIC FE 21/30')
    train2['sentence_std'] = train2.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: np.std(x))
    test2['sentence_std'] = test2.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: np.std(x))

    print_step('BASIC FE 22/30')
    train2['words_per_sentence'] = train2['num_words'] / train2['num_sentence']
    test2['words_per_sentence'] = test2['num_words'] / test2['num_sentence']

    print_step('BASIC FE 23/30')
    train2['num_repeated_sentences'] = train2['sentence'].apply(lambda x: len(x) - len(set(x)))
    test2['num_repeated_sentences'] = test2['sentence'].apply(lambda x: len(x) - len(set(x)))
    train2.drop('sentence', inplace=True, axis=1)
    test2.drop('sentence', inplace=True, axis=1)

    # From https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram
    print_step('BASIC FE 24/30')
    train2['start_with_columns'] = train2['comment_text'].apply(lambda x: 1 if re.search(r'^\:+', x) else 0)
    test2['start_with_columns'] = test2['comment_text'].apply(lambda x: 1 if re.search(r'^\:+', x) else 0)

    print_step('BASIC FE 25/30')
    train2['has_timestamp'] = train2['comment_text'].apply(lambda x: 1 if re.search(r'\d{2}|:\d{2}', x) else 0)
    test2['has_timestamp'] = test2['comment_text'].apply(lambda x: 1 if re.search(r'\d{2}|:\d{2}', x) else 0)

    print_step('BASIC FE 26/30')
    train2['has_date_long'] = train2['comment_text'].apply(lambda x: 1 if re.search(r'\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}', x) else 0)
    test2['has_date_long'] = test2['comment_text'].apply(lambda x: 1 if re.search(r'\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}', x) else 0)

    print_step('BASIC FE 27/30')
    train2['has_date_short'] = train2['comment_text'].apply(lambda x: 1 if re.search(r'\D\d{1,2} \w+ \d{4}', x) else 0)
    test2['has_date_short'] = test2['comment_text'].apply(lambda x: 1 if re.search(r'\D\d{1,2} \w+ \d{4}', x) else 0)

    print_step('BASIC FE 28/30')
    train2['has_link'] = train2['comment_text'].apply(lambda x: 1 if re.search(r'http[s]{0,1}://\S+', x) else (1 if re.search(r'www\.\S+', x) else 0))
    test2['has_link'] = test2['comment_text'].apply(lambda x: 1 if re.search(r'http[s]{0,1}://\S+', x) else (1 if re.search(r'www\.\S+', x) else 0))

    print_step('BASIC FE 29/30')
    train2['has_email'] = train2['comment_text'].apply(lambda x: 1 if re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x) else 0)
    test2['has_email'] = test2['comment_text'].apply(lambda x: 1 if re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x) else 0)

    print_step('BASIC FE 30/30')
    train2['has_ip_address'] = train2['comment_text'].apply(lambda x: 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', x) else 0)
    test2['has_ip_address'] = test2['comment_text'].apply(lambda x: 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', x) else 0)


    ## PART OF SPEECH
    print_step('POS 1/26')
    train2['pos_space'] = train2.comment_text.apply(lambda t: [x for x in TextBlob(normalize_text(t)).pos_tags])
    print_step('POS 2/26')
    test2['pos_space'] = test2.comment_text.apply(lambda t: [x for x in TextBlob(normalize_text(t)).pos_tags])
    print_step('POS 3/26')
    train2['has_foreign_word'] = train2['pos_space'].apply(lambda ws: any(['FW' in w[1] for w in ws]))
    test2['has_foreign_word'] = test2['pos_space'].apply(lambda ws: any(['FW' in w[1] for w in ws]))
    print_step('POS 4/26')
    train2['num_noun'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'NN' in w[1]]))
    test2['num_noun'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'NN' in w[1]]))
    print_step('POS 5/26')
    train2['noun_per_word'] = train2['num_noun'] / train2['num_words']
    test2['noun_per_word'] = test2['num_noun'] / test2['num_words']
    print_step('POS 6/26')
    train2['num_conjunction'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'CC' in w[1]]))
    test2['num_conjunction'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'CC' in w[1]]))
    print_step('POS 7/26')
    train2['num_determiner'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'DT' in w[1]]))
    test2['num_determiner'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'DT' in w[1]]))
    print_step('POS 8/26')
    train2['num_preposition'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'IN' in w[1]]))
    test2['num_preposition'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'IN' in w[1]]))
    print_step('POS 9/26')
    train2['num_adjective'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'JJ' in w[1]]))
    test2['num_adjective'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'JJ' in w[1]]))
    print_step('POS 10/26')
    train2['adjective_per_word'] = train2['num_adjective'] / train2['num_words']
    test2['adjective_per_word'] = test2['num_adjective'] / test2['num_words']
    print_step('POS 11/26')
    train2['num_modal'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'MD' in w[1]]))
    test2['num_modal'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'MD' in w[1]]))
    print_step('POS 12/26')
    train2['num_personal_pronoun'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'PRP' in w[1]]))
    test2['num_personal_pronoun'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'PRP' in w[1]]))
    print_step('POS 13/26')
    train2['personal_pronoun_per_word'] = train2['num_personal_pronoun'] / train2['num_words']
    test2['personal_pronoun_per_word'] = test2['num_personal_pronoun'] / test2['num_words']
    print_step('POS 14/26')
    train2['num_adverb'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'RB' in w[1]]))
    test2['num_adverb'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'RB' in w[1]]))
    print_step('POS 15/26')
    train2['num_adverb_participle'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'RP' in w[1]]))
    test2['num_adverb_participle'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'RP' in w[1]]))
    print_step('POS 16/26')
    train2['num_verb'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'VB' in w[1]]))
    test2['num_verb'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'VB' in w[1]]))
    print_step('POS 17/26')
    train2['verb_per_word'] = train2['num_verb'] / train2['num_words']
    test2['verb_per_word'] = test2['num_verb'] / test2['num_words']
    print_step('POS 18/26')
    train2['num_past_verb'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'VBD' in w[1]]))
    test2['num_past_verb'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'VBD' in w[1]]))
    print_step('POS 19/26')
    train2['num_third_singular_present_verb'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'VBZ' in w[1]]))
    test2['num_third_singular_present_verb'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'VBZ' in w[1]]))
    print_step('POS 20/26')
    train2['num_non_third_singular_present_verb'] = train2['pos_space'].apply(lambda ws: len([w for w in ws if 'VBP' in w[1]]))
    test2['num_non_third_singular_present_verb'] = test2['pos_space'].apply(lambda ws: len([w for w in ws if 'VBP' in w[1]]))
    print_step('POS 21/26')
    train2['num_modal_then_verb'] = train2['pos_space'].apply(lambda ws: 'MD VB' in ' '.join([w[1] for w in ws]))
    test2['num_modal_then_verb'] = test2['pos_space'].apply(lambda ws: 'MD VB' in ' '.join([w[1] for w in ws]))
    print_step('POS 22/26')
    train2['has_personal_pronoun_then_singular_present_verb'] = train2['pos_space'].apply(lambda ws: 'PRP VBP' in ' '.join([w[1] for w in ws]))
    test2['has_personal_pronoun_then_singular_present_verb'] = test2['pos_space'].apply(lambda ws: 'PRP VBP' in ' '.join([w[1] for w in ws]))
    print_step('POS 23/26')
    train2['has_adjective_then_noun'] = train2['pos_space'].apply(lambda ws: 'JJ NN' in ' '.join([w[1] for w in ws]))
    test2['has_adjective_then_noun'] = test2['pos_space'].apply(lambda ws: 'JJ NN' in ' '.join([w[1] for w in ws]))
    print_step('POS 24/26')
    train2['has_noun_then_preposition'] = train2['pos_space'].apply(lambda ws: 'NN IN' in ' '.join([w[1] for w in ws]))
    test2['has_noun_then_preposition'] = test2['pos_space'].apply(lambda ws: 'NN IN' in ' '.join([w[1] for w in ws]))
    print_step('POS 25/26')
    train2['has_verb_then_you'] = train2['pos_space'].apply(has_verb_you)
    test2['has_verb_then_you'] = test2['pos_space'].apply(has_verb_you)
    print_step('POS 26/26')
    train2['has_you_then_verb'] = train2['pos_space'].apply(has_you_verb)
    test2['has_you_then_verb'] = test2['pos_space'].apply(has_you_verb)
    train2.drop('pos_space', inplace=True, axis=1)
    test2.drop('pos_space', inplace=True, axis=1)


    # SYLLABLE DATA
    print_step('SYLLABLE 1/10')
    train2['syllable'] = train2['comment_text'].apply(lambda x: [textstat.syllable_count(normalize_text(w)) for w in str(x).split()])
    print_step('SYLLABLE 2/10')
    test2['syllable'] = test2['comment_text'].apply(lambda x: [textstat.syllable_count(normalize_text(w)) for w in str(x).split()])
    print_step('SYLLABLE 3/10')
    train2['syllable_sum'] = train2.syllable.apply(lambda x: sum(x) if len(x) > 0 else 0)
    test2['syllable_sum'] = test2.syllable.apply(lambda x: sum(x) if len(x) > 0 else 0)
    print_step('SYLLABLE 4/10')
    train2['syllable_mean'] = train2.syllable.apply(lambda x: np.mean(x))
    test2['syllable_mean'] = test2.syllable.apply(lambda x: np.mean(x))
    print_step('SYLLABLE 5/10')
    train2['syllable_max'] = train2.syllable.apply(lambda x: max(x) if len(x) > 0 else 0)
    test2['syllable_max'] = test2.syllable.apply(lambda x: max(x) if len(x) > 0 else 0)
    print_step('SYLLABLE 6/10')
    train2['syllable_std'] = train2.syllable.apply(lambda x: np.std(x))
    test2['syllable_std'] = test2.syllable.apply(lambda x: np.std(x))
    print_step('SYLLABLE 7/10')
    train2['num_big_words'] = train2.syllable.apply(lambda xs: len([x for x in xs if x > 2]))
    test2['num_big_words'] = test2.syllable.apply(lambda xs: len([x for x in xs if x > 2]))
    print_step('SYLLABLE 8/10')
    train2['num_simple_words'] = train2.syllable.apply(lambda xs: len([x for x in xs if x == 1]))
    test2['num_simple_words'] = test2.syllable.apply(lambda xs: len([x for x in xs if x == 1]))
    print_step('SYLLABLE 9/10')
    train2['syllable_per_word'] = train2['syllable_sum'] / (train2['num_words'] + 0.0001)
    test2['syllable_per_word'] = test2['syllable_sum'] / (test2['num_words'] + 0.0001)
    print_step('SYLLABLE 10/10')
    train2['big_words_per_word'] = train2['num_big_words'] / (train2['num_words'] + 0.0001)
    test2['big_words_per_word'] = test2['num_big_words'] / (test2['num_words'] + 0.0001)
    train2.drop('syllable', inplace=True, axis=1)
    test2.drop('syllable', inplace=True, axis=1)


    # READABILITY
    # https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    # https://en.wikipedia.org/wiki/SMOG
    # https://en.wikipedia.org/wiki/Linsear_Write
    # https://en.wikipedia.org/wiki/Automated_readability_index
    # https://en.wikipedia.org/wiki/Gunning_fog_index
    print_step('READABILITY 1/6')
    train2['FRE'] = 206.835 - 1.015 * train2['words_per_sentence'] - 84.6 * train2['syllable_per_word']
    test2['FRE'] = 206.835 - 1.015 * test2['words_per_sentence'] - 84.6 * test2['syllable_per_word']
    print_step('READABILITY 2/6')
    train2['FKGLF'] = 0.39 * train2['words_per_sentence'] + 11.8 * train2['syllable_per_word'] - 15.99
    test2['FKGLF'] = 0.39 * test2['words_per_sentence'] + 11.8 * test2['syllable_per_word'] - 15.99
    print_step('READABILITY 3/6')
    train2['SMOG'] = 1.0430 * train2['num_big_words'] ** 0.5 + 3.1291
    test2['SMOG'] = 1.0430 * test2['num_big_words'] ** 0.5 + 3.1291
    print_step('READABILITY 4/6')
    train2['LW'] = (2 * train2['num_big_words'] + train2['num_words']) / train2['num_sentence']
    test2['LW'] = (2 * test2['num_big_words'] + test2['num_words']) / test2['num_sentence']
    print_step('READABILITY 5/6')
    train2['ARI'] = 4.71 * train2['chars_per_word'] + 0.5 * train2['words_per_sentence'] - 21.43
    test2['ARI'] = 4.71 * test2['chars_per_word'] + 0.5 * test2['words_per_sentence'] - 21.43
    print_step('READABILITY 6/6')
    train2['GFI'] = 0.4 * (train2['words_per_sentence'] + 100 * train2['big_words_per_word'])
    test2['GFI'] = 0.4 * (test2['words_per_sentence'] + 100 * test2['big_words_per_word'])


    # BAD WORDS
    # https://kaggle2.blob.core.windows.net/forum-message-attachments/4810/badwords.txt
    # https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/lexicons/hatebase_dict.csv
    # Top words from vectorizers
    insult = ['ahole', 'dumbass', r'\bass', 'azzhole', 'basterd', 'bitch', 'biatch', 'muncher', 'cawk',
              'whore', 'jackoff', 'jerk', 'jerkoff', 'troll', r'tard\b', 'prick', 'pusse', 'pussy',
              'p\*ssy', 'skank', 'slut', r'\bfat\b', 'nutsack', r'pig\b', 'pigs', 'piggy', 'wuss',
              r'\bspic\b', 'b\*tch', 'idiot', 'stupid', 'ugly', 'facist', 'moron', 'ignorant',
              r'\bworm\b', 'lazy', 'loser', 'noob', r'\bcow', r'\bfool\b', 'ignoramus', 'imbecile',
              'dumb', 'pansy', 'pansies', 'hillbilly', 'trailer trash', 'wetback', 'twat',
              'piece of garbage', 'commie', 'pedo', 'peado', 'douche', 'doushe', 'dousch',
              'monkey', 'boi', 'slime', 'inbred', 'maggot', 'weakling', 'worthless', 'disgrace',
			  'pompous', 'pathetic', 'nerd']
    n_word = [r'\bnig\b', 'n\*g', 'nigg', 'chink', 'chynk', 'nig nog', 'wigger', 'wigga',
              r'\bjigg']
    mean_identity = ['dyke', 'dike', 'fag', 'gay', 'tranny', 'lesbo', 'queer', 'qweer', 'shemale', r'\bnegro',
                     r'\bgyp', r'\bgips', 'white trash', 'darkie', 'chinaman', r'\bcracker\b', 'eurotrash', r'\bsperg', 'aspie',
                     r'\bjap\b', 'christ killer', r'\bhomo\b', 'redneck', 'jew-hat', 'jew hat',
                     r'\bturk\b', 'hermaphrodite', r'\bretard\b', 'subhuman']
    identity = ['lesbian', 'bisexual', r'\btrans\b', 'transex', 'lesbian', r'jew\b', 'jewish', 'black', 'arab',
                'muslim', 'asian', 'islam', 'jihad', 'mexican', 'african', 'irish', 'homosexual',
                'frenchman', 'niger', 'transvest', 'afghan', 'armenian', 'speak english', 'semit',
                'christian', 'chinese', 'manchu', 'white', 'indian', 'women', r'\bnation\b']
    severe_toxic = ['c0ck', 'cock', 'cawk', r'clit\b', 'cunt', 'c\*nt', r'\bcum\b', 'fuc', 'fuk']
    toxic = ['crap', 'blowjob', 'blows me', 'blow me', 'go blow', 'butthole', 'buttwipe', 'dick',
            'dildo', r'\bf u\b', 'fart', 'gook', 'damn', r'goddam\b', r'f.ck', 'f\*\*', 'di\*k',
            'a$$', 'masturb', 'shit', 'sh1t', 'sh!t', r'\btit[^-][^f]', 'turd', 'arse', 'balls',
            'suck', 'fcuk', 'mofo', 'nazi', 'phuck', 'boob', 'b00b', 'wank', '@ss', '@$$',
            'breast', 'cabron', 'feces', 'poop', 'porn', 'p0rn', 'screw', r'\bf you\b',
            r'\bstfu\b', 'a-hole', 'wtf', 'rubbish', r'\banal\b', r'\bpeto', 'prostitut',
            'shut up', 'scum', 'thug', 'filth', 'rapist', 'bagger', 'communist', 'racist', 'hate',
            'sodomize', 'mother f ', 'lowly', 'cancer', 'spews', r'\bhag\b', 'tr0ll', r'\bha\b',
			'haha', 'cry', 'lover', 'loving', 'sick', 'smell', 'freak', r'basement\b', 'anus']
    obscene = ['jack off', 'jizz', 'penis', 'semen', 'vagina', 'ejaculate', 'testicle', 'vajay',
			   'piss', 'pecker', 'urinate']
    wiki = ['wiki', 'jimbo', 'talk', 'page', 'edit', 'signed', 'block', 'edit', 'contribute', 'bot',
            'image:', 'file:', 'deletion', 'user:', '==', r'\baccount\b', 'sock puppet', 'sockpuppet',
            'barnstar', 'tutorial', 'source']
    threat = ['bludgeon', 'beat', 'to death', r'\bshoot', 'die', r'\bshot', 'dead', 'kill', 'punch',
              'burn', 'blood', 'destroy', r'\bcut\b', 'rape', 'stab', r'knock you\b', 'kick', 'hunt you', 'abuse',
              r'happen to you\b', 'better watch', 'suicide', 'execute', 'blow up', r'\bso watch',
              'been warned', 'where you live', 'attack', 'warning', 'track you down', 'hope u die',
              'hope you die', 'skin you', 'infadel', r'\bchoke\b', r'\broast\b', 'succumb', 'corpse', 'agony',
			  'stomp', r'\bsever\b', 'painful', 'petrol', 'snitch', 'shotgun', 'kidnap', 'smack', 'scream',
			  'guts', 'bleed', r'\bshove\b', r'\bsmash\b', 'stalker', 'stalking', r'\bhang\b', r'\bdrown\b',
			  r'\bslit\b', 'castrate', 'gonna', r'\brot\b']
    objects = ['you', 'my', r'\bur\b', 'urself', r'\bu r\b', r'\bmum', r'mom\b', r'\bmother\b', r'\bis a\b', r'\bis an\b',
			  r'\bgo\b', 'mutha']
    self = [r'\bme\b', r'\bi\b', r'\bmy\b', 'myself']
    qualifiers = ['offensive', 'derogatory']
    polite = ['thank', 'help', 'please', 'welcome']

    count = 0
    total = len(insult) + len(n_word) + len(mean_identity) + len(identity) + len(severe_toxic) + len(toxic) + \
            len(obscene) + len(wiki) + len(threat) + len(objects) + len(qualifiers) + len(self) + len(polite) + 15

    for i in insult:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_insult'] = train2[['has_' + str(i) for i in insult]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_insult'] = test2[['has_' + str(i) for i in insult]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in n_word:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_n_word'] = train2[['has_' + str(i) for i in n_word]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_n_word'] = test2[['has_' + str(i) for i in n_word]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in mean_identity:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_mean_identity'] = train2[['has_' + str(i) for i in mean_identity]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_mean_identity'] = test2[['has_' + str(i) for i in mean_identity]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in identity:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_identity'] = train2[['has_' + str(i) for i in identity]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_identity'] = test2[['has_' + str(i) for i in identity]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in severe_toxic:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_severe_toxic'] = train2[['has_' + str(i) for i in severe_toxic]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_severe_toxic'] = test2[['has_' + str(i) for i in severe_toxic]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in toxic:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_toxic'] = train2[['has_' + str(i) for i in toxic]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_toxic'] = test2[['has_' + str(i) for i in toxic]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in obscene:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_obscene'] = train2[['has_' + str(i) for i in obscene]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_obscene'] = test2[['has_' + str(i) for i in obscene]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in wiki:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_wiki'] = train2[['has_' + str(i) for i in wiki]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_wiki'] = test2[['has_' + str(i) for i in wiki]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in threat:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_threat'] = train2[['has_' + str(i) for i in threat]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_threat'] = test2[['has_' + str(i) for i in threat]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in objects:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_objects'] = train2[['has_' + str(i) for i in objects]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_objects'] = test2[['has_' + str(i) for i in objects]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in self:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_self'] = train2[['has_' + str(i) for i in self]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_self'] = test2[['has_' + str(i) for i in self]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in qualifiers:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_qualifiers'] = train2[['has_' + str(i) for i in qualifiers]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_qualifiers'] = test2[['has_' + str(i) for i in qualifiers]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in polite:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train2['has_' + str(i)] = train2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test2['has_' + str(i)] = test2.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['has_cat_polite'] = train2[['has_' + str(i) for i in polite]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test2['has_cat_polite'] = test2[['has_' + str(i) for i in polite]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['num_bad_words'] = train2[[c for c in train2.columns if 'has' in c and 'has_cat' not in c]].sum(axis=1) - train2[['has_' + str(i) for i in qualifiers] + ['has_' + str(i) for i in wiki] + ['has_' + str(i) for i in objects] + ['has_' + str(i) for i in polite] + ['has_' + str(i) for i in self]].sum(axis=1)
    test2['num_bad_words'] = test2[[c for c in test2.columns if 'has' in c and 'has_cat' not in c]].sum(axis=1) - test2[['has_' + str(i) for i in qualifiers] + ['has_' + str(i) for i in wiki] + ['has_' + str(i) for i in objects] + ['has_' + str(i) for i in polite] + ['has_' + str(i) for i in self]].sum(axis=1)

    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train2['bad_words_per_word'] = train2['num_bad_words'] / train2['num_words']
    test2['bad_words_per_word'] = test2['num_bad_words'] / test2['num_words']


    ## AFINN
    afinn = Afinn()
    print_step('AFINN 1/14')
    train2['afinn'] = train2.comment_text.apply(lambda xs: [afinn.score(x) for x in xs.split()])
    print_step('AFINN 2/14')
    test2['afinn'] = test2.comment_text.apply(lambda xs: [afinn.score(x) for x in xs.split()])
    print_step('AFINN 3/14')
    train2['afinn_sum'] = train2.afinn.apply(lambda x: sum(x) if len(x) > 0 else 0)
    test2['afinn_sum'] = test2.afinn.apply(lambda x: sum(x) if len(x) > 0 else 0)
    print_step('AFINN 4/14')
    train2['afinn_mean'] = train2.afinn.apply(lambda x: np.mean(x))
    test2['afinn_mean'] = test2.afinn.apply(lambda x: np.mean(x))
    print_step('AFINN 5/14')
    train2['afinn_max'] = train2.afinn.apply(lambda x: max(x) if len(x) > 0 else 0)
    test2['afinn_max'] = test2.afinn.apply(lambda x: max(x) if len(x) > 0 else 0)
    print_step('AFINN 6/14')
    train2['afinn_min'] = train2.afinn.apply(lambda x: min(x) if len(x) > 0 else 0)
    test2['afinn_min'] = test2.afinn.apply(lambda x: min(x) if len(x) > 0 else 0)
    print_step('AFINN 7/14')
    train2['afinn_std'] = train2.afinn.apply(lambda x: np.std(x))
    test2['afinn_std'] = test2.afinn.apply(lambda x: np.std(x))
    print_step('AFINN 8/14')
    train2['afinn_num'] = train2.afinn.apply(lambda xs: len([x for x in xs if x != 0]))
    test2['afinn_num'] = test2.afinn.apply(lambda xs: len([x for x in xs if x != 0]))
    print_step('AFINN 9/14')
    train2['afinn_num_pos'] = train2.afinn.apply(lambda xs: len([x for x in xs if x > 0]))
    test2['afinn_num_pos'] = test2.afinn.apply(lambda xs: len([x for x in xs if x > 0]))
    print_step('AFINN 10/14')
    train2['afinn_num_neg'] = train2.afinn.apply(lambda xs: len([x for x in xs if x < 0]))
    test2['afinn_num_neg'] = test2.afinn.apply(lambda xs: len([x for x in xs if x < 0]))
    print_step('AFINN 11/14')
    train2['afinn_per_word'] = train2['afinn_num'] / (train2['num_words'] + 0.0001)
    test2['afinn_per_word'] = test2['afinn_num'] / (test2['num_words'] + 0.0001)
    print_step('AFINN 12/14')
    train2['afinn_pos_per_word'] = train2['afinn_num_pos'] / (train2['num_words'] + 0.0001)
    test2['afinn_pos_per_word'] = test2['afinn_num_pos'] / (test2['num_words'] + 0.0001)
    print_step('AFINN 13/14')
    train2['afinn_neg_per_word'] = train2['afinn_num_neg'] / (train2['num_words'] + 0.0001)
    test2['afinn_neg_per_word'] = test2['afinn_num_neg'] / (test2['num_words'] + 0.0001)
    print_step('AFINN 14/14')
    train2['afinn_neg_per_pos'] = train2['afinn_num_pos'] / (train2['afinn_num_neg'] + 0.0001)
    test2['afinn_neg_per_pos'] = test2['afinn_num_pos'] / (test2['afinn_num_neg'] + 0.0001)
    train2.drop('afinn', inplace=True, axis=1)
    test2.drop('afinn', inplace=True, axis=1)


    ## SENTIMENT
    print_step('Sentiment 1/2')
    train2['sentiment'] = train2.comment_text.apply(lambda t: TextBlob(normalize_text(t)).sentiment.polarity)
    print_step('Sentiment 2/2')
    test2['sentiment'] = test2.comment_text.apply(lambda t: TextBlob(normalize_text(t)).sentiment.polarity)


    print('Train shape: {}'.format(train2.shape))
    print('Test shape: {}'.format(test2.shape))
    return train2, test2
