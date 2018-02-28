import re
import string

from nltk.corpus import stopwords

from utils import print_step


stopwords = {x: 1 for x in stopwords.words('english')}

def add_features(train, test):
    print_step('Feature engineering 1/13')
    train['num_words'] = train['comment_text'].apply(lambda x: len(str(x).split()))
    test['num_words'] = test['comment_text'].apply(lambda x: len(str(x).split()))

    print_step('Feature engineering 2/13')
    train['num_unique_words'] = train['comment_text'].apply(lambda x: len(set(str(x).lower().split())))
    test['num_unique_words'] = test['comment_text'].apply(lambda x: len(set(str(x).lower().split())))

    print_step('Feature engineering 3/13')
    train['num_chars'] = train['comment_text'].apply(lambda x: len(str(x)))
    test['num_chars'] = test['comment_text'].apply(lambda x: len(str(x)))

    print_step('Feature engineering 4/13')
    train['num_capital'] = train['comment_text'].apply(lambda x: len([c for c in x if c.isupper()]))
    test['num_capital'] = test['comment_text'].apply(lambda x: len([c for c in x if c.isupper()]))

    print_step('Feature engineering 5/13')
    train['num_lowercase'] = train['comment_text'].apply(lambda x: len([c for c in x if c.islower()]))
    test['num_lowercase'] = test['comment_text'].apply(lambda x: len([c for c in x if c.islower()]))

    print_step('Feature engineering 6/13')
    train['capital_per_char'] = train['num_capital'] / train['num_chars']
    test['capital_per_char'] = test['num_capital'] / test['num_chars']

    print_step('Feature engineering 7/13')
    train['lowercase_per_char'] = train['num_lowercase'] / train['num_chars']
    test['lowercase_per_char'] = test['num_lowercase'] / test['num_chars']

    print_step('Feature engineering 8/13')
    train['num_stopwords'] = train['comment_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))
    test['num_stopwords'] = test['comment_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))

    print_step('Feature engineering 9/13')
    train['num_punctuations'] = train['comment_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    test['num_punctuations'] = test['comment_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    print_step('Feature engineering 10/13')
    train['punctuation_per_char'] = train['num_punctuations'] / train['num_chars']
    test['punctuation_per_char'] = test['num_punctuations'] / test['num_chars']

    print_step('Feature engineering 11/13')
    train['num_words_upper'] = train['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    test['num_words_upper'] = test['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    print_step('Feature engineering 12/13')
    train['num_words_lower'] = train['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
    test['num_words_lower'] = test['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))

    print_step('Feature engineering 13/13')
    train['num_words_title'] = train['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    test['num_words_title'] = test['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    print('Train shape: {}'.format(train.shape))
    print('Test shape: {}'.format(test.shape))
    return train, test
