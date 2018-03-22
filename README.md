## Kaggle Jigsaw Toxic Comment, 34th / 4551 (Top 1%) Solution

See a detailed methodology at https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52645.

~~

To reproduce the final solution, download data from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data and put it in `data` folder, make a `cache` and `submit` folder. Download Neptune models from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/51836. Zafar's cleaned data comes from https://www.kaggle.com/fizzbuzz/toxic-data-preprocessing/code. You also will have to download embedding files. (See more `setup.txt`.)

 Run `pip install -r requirements.txt`, and then run all the models:

```
python lr.py
python fe_lgb.py
python sparse_lgb.py
python sparse_fe_lgb.py
python gru.py
python gru2.py
python gru80.py
python gru128.py
python gru128-2.py
python gru-conv.py
python double-gru.py
python 2dconv.py
python lstm-conv.py
python dpcnn.py
python rnncnn.py
python rnncnn2.py
python capsule_net.py
python attention-lstm.py
python fm.py
python ridge.py
python neptune-ml-models.py
python lvl2_final_cnn.py
python lvl2_final_lgb.py
```

After that, you will have a final submission ready to go. `cache.py` and `cv.py` maintain the model run layer and are imported, not called directly. `feature_engineering.py` contains all the feature engineering, and is imported and run as needed. `preprocess.py` and `utils.py` contain additional functions that are used by the models.

`lvl3_hillclimb_average.py` should be run after `python lvl2_final_lgb.py` to find optimal weights. These weights are then manually normalized to sum up to 1 and then manually put back into `python lvl2_final_lgb.py`.

`relative_word_frequency_analysis.py` identifies words that are more frequently found in toxic than non-toxic comments. It was used to inform the word list found in `feature_engineering.py`.

`conversationai_api.py` contains our attempt to query the PerspectiveAPI for data. This ended up being unhelpful. To run this, you will need to set up API access.

`lr-extra_data.py` contains our attempt to run linear regressions on the extra Wikipedia toxicity and agression labels. This ended up being unhelpful.
