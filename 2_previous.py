import pandas as pd
import numpy as np
import re
from setting import *


def featurization(df):
    df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    df['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

    df['APP_CREDIT_PERC'] = df['AMT_APPLICATION'] / df['AMT_CREDIT']
    df_numerical = df.groupby('SK_ID_CURR').agg({
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    })
    df_numerical.columns = pd.Index([re.sub('[^0-9A-Z_]+', '_', (e[0] + "_" + e[1]).upper()) for e in df_numerical.columns.tolist()])

    df_numerical_approved = df[df.NAME_CONTRACT_STATUS == 'Approved'].groupby('SK_ID_CURR').agg({
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    })
    df_numerical_approved.columns = pd.Index([re.sub('[^0-9A-Z_]+', '_', ("APR_" + e[0] + "_" + e[1]).upper()) for e in df_numerical_approved.columns.tolist()])

    df_numerical_rejected = df[df.NAME_CONTRACT_STATUS == 'Refused'].groupby('SK_ID_CURR').agg({
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    })
    df_numerical_rejected.columns = pd.Index([re.sub('[^0-9A-Z_]+', '_', ("REJ_" + e[0] + "_" + e[1]).upper()) for e in df_numerical_rejected.columns.tolist()])

    df_numerical = df_numerical.merge(df_numerical_approved, left_index=True, right_index=True, how='left').\
                                merge(df_numerical_rejected, left_index=True, right_index=True, how='left')

    dfs = []
    for col in df.columns:
        if df[col].dtype == 'object':
            t = pd.crosstab(df.SK_ID_CURR, df[col].fillna('NaN'), normalize='index')
            t.columns = pd.Index([re.sub('[^0-9A-Z_]+', '_', (t.columns.name + '_' + itm).upper()) for itm in t.columns.tolist()])
            dfs.append(t)
    df_categorial = pd.concat(dfs, axis = 1)
    return pd.concat([df_numerical, df_categorial], axis = 1).add_prefix('PREV_')

previous = pd.read_csv(BASEDIR + '/previous_application.csv', engine='c')
rst = featurization(previous)
train = pd.read_csv(BASEDIR + '/train/application.csv', engine='c')
train[['SK_ID_CURR']].merge(rst, left_on='SK_ID_CURR', right_index=True).to_csv(BASEDIR+'/train/previous.csv', index=False)
test = pd.read_csv(BASEDIR + '/test/application.csv', engine='c')
test[['SK_ID_CURR']].merge(rst, left_on='SK_ID_CURR', right_index=True).to_csv(BASEDIR+'/test/previous.csv', index=False)