import re
import string

from textblob import TextBlob
from afinn import Afinn

from nltk.corpus import stopwords

from utils import print_step


def add_features(train, test):
    # Basic features
    print_step('Basic FE 1/14')
    train['num_words'] = train['comment_text'].apply(lambda x: len(str(x).split()))
    test['num_words'] = test['comment_text'].apply(lambda x: len(str(x).split()))

    print_step('Basic FE 2/14')
    train['num_unique_words'] = train['comment_text'].apply(lambda x: len(set(str(x).lower().split())))
    test['num_unique_words'] = test['comment_text'].apply(lambda x: len(set(str(x).lower().split())))

    print_step('Basic FE 3/14')
    train['unique_words_per_word'] = train['num_unique_words'] / (train['num_words'] + 0.0001)
    test['unique_words_per_word'] = test['num_unique_words'] / (test['num_words'] + 0.0001)

    print_step('Basic FE 4/14')
    train['num_chars'] = train['comment_text'].apply(lambda x: len(str(x)))
    test['num_chars'] = test['comment_text'].apply(lambda x: len(str(x)))

    print_step('Basic FE 5/14')
    train['num_capital'] = train['comment_text'].apply(lambda x: len([c for c in x if c.isupper()]))
    test['num_capital'] = test['comment_text'].apply(lambda x: len([c for c in x if c.isupper()]))

    print_step('Basic FE 6/14')
    train['num_lowercase'] = train['comment_text'].apply(lambda x: len([c for c in x if c.islower()]))
    test['num_lowercase'] = test['comment_text'].apply(lambda x: len([c for c in x if c.islower()]))

    print_step('Basic FE 7/14')
    train['capital_per_char'] = train['num_capital'] / train['num_chars']
    test['capital_per_char'] = test['num_capital'] / test['num_chars']

    print_step('Basic FE 8/14')
    train['lowercase_per_char'] = train['num_lowercase'] / train['num_chars']
    test['lowercase_per_char'] = test['num_lowercase'] / test['num_chars']

    print_step('Basic FE 9/14')
    stopwords = {x: 1 for x in stopwords.words('english')}
    train['num_stopwords'] = train['comment_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))
    test['num_stopwords'] = test['comment_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))

    print_step('Basic FE 10/14')
    train['num_punctuations'] = train['comment_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    test['num_punctuations'] = test['comment_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    print_step('Basic FE 11/14')
    train['punctuation_per_char'] = train['num_punctuations'] / train['num_chars']
    test['punctuation_per_char'] = test['num_punctuations'] / test['num_chars']

    print_step('Basic FE 12/14')
    train['num_words_upper'] = train['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    test['num_words_upper'] = test['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    print_step('Basic FE 13/14')
    train['num_words_lower'] = train['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
    test['num_words_lower'] = test['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))

    print_step('Basic FE 14/14')
    train['num_words_title'] = train['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    test['num_words_title'] = test['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))


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
              'piece of garbage', 'commie']
    n_word = [r'\bnig\b', 'n\*g', 'nigg', 'chink', 'chynk', 'nig nog', 'wigger', 'wigga']
    mean_identity = ['dyke', 'dike', 'fag', 'gay', 'tranny', 'lesbo', 'queer', 'qweer', 'shemale', r'\bnegro',
                     r'\bgyp', r'\bgips', 'white trash', 'darkie', 'chinaman', '\bcracker\b', 'eurotrash', '\bsperg', 'aspie',
                     '\bjap\b', 'christ killer', r'\bhomo\b', 'redneck']
    identity = ['lesbian', 'bisexual', r'\btrans\b', 'transex', 'lesbian', r'jew\b', 'jewish', 'black', 'arab',
                'muslim', 'monkey', 'asian', 'islam', 'jihad', 'mexican', 'african', 'subhuman', 'irish', 'homosexual',
                'frenchman', 'niger', 'transvest', 'afghan', 'armenian', 'speak english']
    severe_toxic = ['c0ck', 'cock', 'cawk', r'clit\b', 'cunt', 'c\*nt', r'\bcum\b', 'fuc', 'fuk']
    toxic = ['crap', 'blowjob', 'blows me', 'blow me', 'go blow', 'butthole', 'buttwipe', 'dick',
            'dildo', r'\bf u\b', 'fart', 'gook', 'damn', r'goddam\b', 'f\*ck', 'f\*\*k', 'di\*k',
            'a$$', 'masturb', 'shit', 'sh1t', 'sh!t', r'\btit[^-][^f]', 'turd', 'arse', 'balls',
            'suck', 'fcuk', 'mofo', 'nazi', 'phuck', 'boob', 'b00b', 'wank', '@ss', '@$$',
            'breast', 'cabron', 'feces', 'poop', 'porn', 'p0rn', 'screw', r'\bf you\b',
            r'\bstfu\b', 'a-hole', 'wtf', 'rubbish', r'\banal\b', r'\bpeto', 'prostitut',
            'shut up', 'scum', 'thug', 'filth', 'rapist', 'bagger', 'communist']
    obscene = ['jack off', 'jizz', 'penis', 'semen', 'vagina', 'ejaculate', 'testicle', 'vajay', 'piss', 'pecker']
    wiki = ['wiki', 'jimbo', 'talk', 'page', 'edit', 'signed', 'block', 'edit', 'contribute']
    threat = ['bludgeon', 'beat', 'to death', r'\bshoot', 'die', r'\bshot', 'dead', 'kill', 'punch',
              'burn', 'blood', 'destroy', r'\bcut\b', 'rape', 'stab', r'knock you\b', 'kick', 'hunt you', 'abuse',
              r'happen to you\b', 'better watch', 'commit suicide', 'execute', 'blow up', r'\bso watch',
              'been warned', 'where you live']
    objects = ['you', 'my', '\bur\b', 'urself', '\bu r\b', r'\bmum', r'mom\b', r'\bmother\b', r'\bis a\b', r'\bis an\b', r'\bgo\b']
    qualifiers = ['offensive', 'derogatory']

    count = 0
    total = len(insult) + len(n_word) + len(mean_identity) + len(identity) + len(severe_toxic) + len(toxic) + \
            len(obscene) + len(wiki) + len(threat) + len(objects) + len(qualifiers) + 13

    for i in insult:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train['has_' + str(i)] = train.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test['has_' + str(i)] = test.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['has_cat_insult'] = train[['has_' + str(i) for i in insult]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test['has_cat_insult'] = test[['has_' + str(i) for i in insult]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in n_word:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train['has_' + str(i)] = train.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test['has_' + str(i)] = test.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['has_cat_n_word'] = train[['has_' + str(i) for i in n_word]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test['has_cat_n_word'] = test[['has_' + str(i) for i in n_word]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in mean_identity:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train['has_' + str(i)] = train.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test['has_' + str(i)] = test.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['has_cat_mean_identity'] = train[['has_' + str(i) for i in mean_identity]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test['has_cat_mean_identity'] = test[['has_' + str(i) for i in mean_identity]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in identity:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train['has_' + str(i)] = train.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test['has_' + str(i)] = test.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['has_cat_identity'] = train[['has_' + str(i) for i in identity]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test['has_cat_identity'] = test[['has_' + str(i) for i in identity]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in severe_toxic:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train['has_' + str(i)] = train.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test['has_' + str(i)] = test.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['has_cat_severe_toxic'] = train[['has_' + str(i) for i in severe_toxic]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test['has_cat_severe_toxic'] = test[['has_' + str(i) for i in severe_toxic]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in toxic:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train['has_' + str(i)] = train.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test['has_' + str(i)] = test.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['has_cat_toxic'] = train[['has_' + str(i) for i in toxic]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test['has_cat_toxic'] = test[['has_' + str(i) for i in toxic]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in obscene:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train['has_' + str(i)] = train.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test['has_' + str(i)] = test.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['has_cat_obscene'] = train[['has_' + str(i) for i in obscene]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test['has_cat_obscene'] = test[['has_' + str(i) for i in obscene]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in wiki:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train['has_' + str(i)] = train.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test['has_' + str(i)] = test.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['has_cat_wiki'] = train[['has_' + str(i) for i in wiki]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test['has_cat_wiki'] = test[['has_' + str(i) for i in wiki]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in threat:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train['has_' + str(i)] = train.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test['has_' + str(i)] = test.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['has_cat_threat'] = train[['has_' + str(i) for i in threat]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test['has_cat_threat'] = test[['has_' + str(i) for i in threat]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in objects:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train['has_' + str(i)] = train.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test['has_' + str(i)] = test.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['has_cat_objects'] = train[['has_' + str(i) for i in objects]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test['has_cat_objects'] = test[['has_' + str(i) for i in objects]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    for i in qualifiers:
        count += 1
        print_step('BAD WORDS ' + str(count) + '/' + str(total) + ': ' + str(i))
        train['has_' + str(i)] = train.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
        test['has_' + str(i)] = test.comment_text.apply(lambda t: 1 if re.search(i, t.lower()) else 0)
    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['has_cat_qualifiers'] = train[['has_' + str(i) for i in qualifiers]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    test['has_cat_qualifiers'] = test[['has_' + str(i) for i in qualifiers]].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['num_bad_words'] = train[[c for c in train.columns if 'has' in c and 'has_cat' not in c]].sum(axis=1) - train[['has_' + str(i) for i in qualifiers] + ['has_' + str(i) for i in wiki] + ['has_' + str(i) for i in objects]].sum(axis=1)
    test['num_bad_words'] = test[[c for c in test.columns if 'has' in c and 'has_cat' not in c]].sum(axis=1) - test[['has_' + str(i) for i in qualifiers] + ['has_' + str(i) for i in wiki] + ['has_' + str(i) for i in objects]].sum(axis=1)

    count += 1; print_step('BAD WORDS ' + str(count) + '/' + str(total))
    train['bad_words_per_word'] = train['num_bad_words'] / train['num_words']
    test['bad_words_per_word'] = test['num_bad_words'] / test['num_words']


    ## AFINN
    afinn = Afinn()
    print_step('AFINN 1/14')
    train['afinn'] = train.comment_text.apply(lambda xs: [afinn.score(x) for x in xs.split()])
    print_step('AFINN 2/14')
    test['afinn'] = test.comment_text.apply(lambda xs: [afinn.score(x) for x in xs.split()])
    print_step('AFINN 3/14')
    train['afinn_sum'] = train.afinn.apply(lambda x: sum(x) if len(x) > 0 else 0)
    test['afinn_sum'] = test.afinn.apply(lambda x: sum(x) if len(x) > 0 else 0)
    print_step('AFINN 4/14')
    train['afinn_mean'] = train.afinn.apply(lambda x: np.mean(x))
    test['afinn_mean'] = test.afinn.apply(lambda x: np.mean(x))
    print_step('AFINN 5/14')
    train['afinn_max'] = train.afinn.apply(lambda x: max(x) if len(x) > 0 else 0)
    test['afinn_max'] = test.afinn.apply(lambda x: max(x) if len(x) > 0 else 0)
    print_step('AFINN 6/14')
    train['afinn_min'] = train.afinn.apply(lambda x: min(x) if len(x) > 0 else 0)
    test['afinn_min'] = test.afinn.apply(lambda x: min(x) if len(x) > 0 else 0)
    print_step('AFINN 7/14')
    train['afinn_std'] = train.afinn.apply(lambda x: np.std(x))
    test['afinn_std'] = test.afinn.apply(lambda x: np.std(x))
    print_step('AFINN 8/14')
    train['afinn_num'] = train.afinn.apply(lambda xs: len([x for x in xs if x != 0]))
    test['afinn_num'] = test.afinn.apply(lambda xs: len([x for x in xs if x != 0]))
    print_step('AFINN 9/14')
    train['afinn_num_pos'] = train.afinn.apply(lambda xs: len([x for x in xs if x > 0]))
    test['afinn_num_pos'] = test.afinn.apply(lambda xs: len([x for x in xs if x > 0]))
    print_step('AFINN 10/14')
    train['afinn_num_neg'] = train.afinn.apply(lambda xs: len([x for x in xs if x < 0]))
    test['afinn_num_neg'] = test.afinn.apply(lambda xs: len([x for x in xs if x < 0]))
    print_step('AFINN 11/14')
    train['afinn_per_word'] = train['afinn_num'] / (train['num_words'] + 0.0001)
    test['afinn_per_word'] = test['afinn_num'] / (test['num_words'] + 0.0001)
    print_step('AFINN 12/14')
    train['afinn_pos_per_word'] = train['afinn_num_pos'] / (train['num_words'] + 0.0001)
    test['afinn_pos_per_word'] = test['afinn_num_pos'] / (test['num_words'] + 0.0001)
    print_step('AFINN 13/14')
    train['afinn_neg_per_word'] = train['afinn_num_neg'] / (train['num_words'] + 0.0001)
    test['afinn_neg_per_word'] = test['afinn_num_neg'] / (test['num_words'] + 0.0001)
    print_step('AFINN 14/14')
    train['afinn_neg_per_pos'] = train['afinn_num_pos'] / (train['afinn_num_neg'] + 0.0001)
    test['afinn_neg_per_pos'] = test['afinn_num_pos'] / (test['afinn_num_neg'] + 0.0001)


    ## SENTIMENT
    non_alphanums = re.compile(u'[^A-Za-z0-9]+')
    def tokenize_fn(text):
        return u" ".join(
            [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(' ')] if len(x) > 1])

    print_step('Sentiment 1/2')
    train['sentiment'] = train.comment_text.apply(lambda t: TextBlob(tokenize_fn(t)).sentiment.polarity)
    print_step('Sentiment 2/2')
    test['sentiment'] = test.comment_text.apply(lambda t: TextBlob(tokenize_fn(t)).sentiment.polarity)

    print('Train shape: {}'.format(train.shape))
    print('Test shape: {}'.format(test.shape))
    return train, test
