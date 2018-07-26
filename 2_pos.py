import pandas as pd
import numpy as np
import re
from setting import *


def featurization(df):
    df_numerical = df.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'SK_ID_CURR': ['size']
    })
    df_numerical.columns = pd.Index([re.sub('[^0-9A-Z_]+', '_', (e[0] + "_" + e[1]).upper()) for e in df_numerical.columns.tolist()])

    dfs = []
    for col in df.columns:
        if df[col].dtype == 'object':
            t = pd.crosstab(df.SK_ID_CURR, df[col].fillna('NaN'), normalize='index')
            t.columns = pd.Index([re.sub('[^0-9A-Z_]+', '_', (t.columns.name + '_' + itm).upper()) for itm in t.columns.tolist()])
            dfs.append(t)
    df_categorial = pd.concat(dfs, axis = 1)
    return pd.concat([df_numerical, df_categorial], axis = 1).add_prefix('POS_')

pos = pd.read_csv(BASEDIR + '/POS_CASH_balance.csv', engine='c')
rst = featurization(pos)
train = pd.read_csv(BASEDIR + '/train/application.csv', engine='c')
train[['SK_ID_CURR']].merge(rst, left_on='SK_ID_CURR', right_index=True).to_csv(BASEDIR+'/train/pos.csv', index=False)
test = pd.read_csv(BASEDIR + '/test/application.csv', engine='c')
test[['SK_ID_CURR']].merge(rst, left_on='SK_ID_CURR', right_index=True).to_csv(BASEDIR+'/test/pos.csv', index=False)