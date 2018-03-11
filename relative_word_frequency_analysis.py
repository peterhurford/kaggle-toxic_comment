import re
from datetime import datetime

# dat = pd.read_csv("data/train.csv")

isnum = re.compile(u'^[0-9]+$')
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

def normalize_text(text):
  return list(set([x for x in [y for y in non_alphanums.sub(" ", text).lower().strip().split(' ')] if len(x) > 1 and not isnum.match(x)]))

def time(fn):
  pre = datetime.now()
  val = fn()
  print(datetime.now() - pre)
  return val

print('Computing bads')
bads = time(lambda: [item for sublist in map(normalize_text, dat['comment_text'][dat['toxic'] == 1]) for item in sublist])
goods = time(lambda: [item for sublist in map(normalize_text, dat['comment_text'][dat['toxic'] == 0]) for item in sublist])
cgoods = Counter(goods)
cbads = Counter(bads)

bad_candidates = cbads.most_common(1500)
bad_candidate_words = map(lambda (word, count): word, bad_candidates)

bad_ratios = map(lambda word: (word, float(cbads[word]) / (cgoods[word] + cbads[word])), bad_candidate_words)
worst_ratios = sorted(bad_ratios, key = lambda (_,y): y, reverse = True)

good_candidates = cgoods.most_common(1500)
good_candidate_words = map(lambda (word, count): word, good_candidates)

good_ratios = map(lambda word: (word, float(cgoods[word]) / (cgoods[word] + cbads[word])), good_candidate_words)
best_ratios = sorted(good_ratios, key = lambda (_,y): y, reverse = True)

arbitrary_bad_threshold = 0.2
arbitrary_good_threshold = 0.98

worst_words = map(lambda x: x[0], filter(lambda (word, ratio): ratio > arbitrary_bad_threshold, worst_ratios))


