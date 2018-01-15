* Get mean log loss metric for level 1 and level 2
* Get and look at feature importance
* Train models for labels other than toxic only on the subset of data that is labeled toxic
* Train models on additional data 
* Add their API as a level 1 model (https://github.com/conversationai/perspectiveapi/blob/master/api_reference.md - I already have access)
* Improve existing models
  * LR https://www.kaggle.com/thousandvoices/logistic-regression-with-words-and-char-n-grams
  * See https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46563
* Additional level 1 models (VW, FTLR, FTLR-FM, XGB, LGB, NN, SVM, LibFM, ExtraTrees, RandomForest)
  * NB-SVM https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb
  * NNs
    * https://www.kaggle.com/demesgal/lstm-glove-lr-decrease-bn-cv-lb-0-047
    * https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043/code
    * https://github.com/AloneGu/kaggle_spooky
* Additional level 2 models
* Additional feature engineering
  * Average length of words
  * Length of longest word
  * LangDetect as a feature https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46330
* Data cleaning?
* Model based on a static blacklist (see https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46035 / https://www.kaggle.com/c/detecting-insults-in-social-commentary/discussion/2744 / https://kaggle2.blob.core.windows.net/forum-message-attachments/4810/badwords.txt)
* Use other external data
  * in data folder - see README / see https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46035
    * BE CAREFUL OF LEAKAGE - SEE https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46177
  * Also use https://conversationai.github.io/wikidetox/testdata/tox-sorted/Wikipedia%20Toxicity%20Sorted%20%28Toxicity%405%5BAlpha%5D%29.html
  * Use data from Kaggle insults competition
  * Data and code from https://github.com/t-davidson/hate-speech-and-offensive-language
  * https://cloud.google.com/natural-language/
* Sentiment analysis
* Part of speech analysis
* Word2Vec, GloVe, etc.
  * https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46736
* Tune model hyperparams (TFIDF, XGB, LR, NB, etc.)
* Review https://github.com/AloneGu/kaggle_spooky
* The entire 95M+ Wikipedia discussion corpus is available if we want to do some sort of big data semi-supervised learning
* https://arxiv.org/pdf/1710.07394.pdf
* https://dl.acm.org/citation.cfm?id=3052591
* https://aws.amazon.com/comprehend/?p=tile
