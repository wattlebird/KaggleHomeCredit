import pandas as pd
import numpy as np
from setting import *


def featurization(df):
    df_numerical = df.drop(columns=['SK_ID_PREV'], axis=1).groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    df_numerical.columns = pd.Index([e[0] + "_" + e[1].upper() for e in df_numerical.columns.tolist()])
    df_count = df[['SK_ID_PREV', 'SK_ID_CURR']].groupby('SK_ID_CURR').SK_ID_PREV.count().add_suffix('_COUNT')
    df_numerical = pd.concat([df_numerical, df_count], axis=1)

    dfs = []
    for col in df.columns:
        if df[col].dtype == 'object':
            t = pd.crosstab(df.SK_ID_CURR, df[col].fillna('NaN'), normalize='index')
            t.columns = pd.Index([t.columns.name + '_' + itm.upper() for itm in t.columns.tolist()])
            dfs.append(t)
    df_categorial = pd.concat(dfs, axis = 1)
    return pd.concat([df_numerical, df_categorial], axis = 1).add_prefix('CREDITCARD_')

cc = pd.read_csv(BASEDIR + '/credit_card_balance.csv', engine='c')
rst = featurization(cc)
train = pd.read_csv(BASEDIR + '/train/application.csv', engine='c')
train[['SK_ID_CURR']].merge(rst, left_on='SK_ID_CURR', right_index=True).to_csv(BASEDIR+'/train/creditcard.csv', index=False)
test = pd.read_csv(BASEDIR + '/test/application.csv', engine='c')
test[['SK_ID_CURR']].merge(rst, left_on='SK_ID_CURR', right_index=True).to_csv(BASEDIR+'/test/creditcard.csv', index=False)