import pandas as pd
import numpy as np
import lightgbm as lgb
from setting import *

train = pd.read_csv(BASEDIR + '/train/train.csv', engine='c')
test = pd.read_csv(BASEDIR + '/test/test.csv', engine='c')
feats = [f for f in train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
for col in train.columns:
    if train[col].dtype=='object':
        train[col] = train[col].astype('category')
for col in test.columns:
    if test[col].dtype=='object':
        test[col] = test[col].astype('category')

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 20,
    'min_data_in_leaf': 640,
    'learning_rate': 0.1,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'scale_pos_weight':10,
    'metric_freq': 5,
    'is_provide_training_metric': True,
    'verbose': 0,
    'num_threads': 4
}

train_x, train_y = train[feats], train['TARGET']
test_x = test[feats]
lgb_train = lgb.Dataset(train_x, train_y)
lgbm = lgb.train({**params}, lgb_train, num_boost_round=213, verbose_eval=False)
pred_y = lgbm.predict(test_x)

result = pd.DataFrame(data={
    'SK_ID_CURR': test.SK_ID_CURR,
    'TARGET': pred_y
}, columns=['SK_ID_CURR', 'TARGET'])

result.to_csv(BASEDIR + '/submission_20180723.csv', index=False)