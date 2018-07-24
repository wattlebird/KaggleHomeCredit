import pandas as pd
import numpy as np
from setting import *


def balance_featurization(df):
    bb_status = pd.crosstab(df.SK_ID_BUREAU, df.STATUS, normalize="index")
    bb_status.columns = pd.Index([bb_status.columns.name + '_' + itm.upper() for itm in bb_status.columns.tolist()])

    bb_month_balance = df.groupby('SK_ID_BUREAU').MONTHS_BALANCE.agg(['min', 'max', 'size'])
    bb_month_balance.columns.name = 'MONTHS_BALANCE'
    bb_month_balance.columns = pd.Index([bb_month_balance.columns.name + '_' + itm.upper() for itm in bb_month_balance.columns.tolist()])

    rtn = pd.concat([bb_status, bb_month_balance], axis=1).add_prefix('BALANCE_')
    return rtn

def featurization(bureau, bb):
    bureau_balance = balance_featurization(bb)
    bureau_balance = bureau[['SK_ID_CURR','SK_ID_BUREAU']].merge(bureau_balance, how='left', left_on='SK_ID_BUREAU', right_index=True)
    bureau_balance.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    bureau_balance = bureau_balance.groupby('SK_ID_CURR').agg({
        'BALANCE_STATUS_0': ['mean'],
        'BALANCE_STATUS_1': ['mean'],
        'BALANCE_STATUS_2': ['mean'],
        'BALANCE_STATUS_3': ['mean'],
        'BALANCE_STATUS_4': ['mean'],
        'BALANCE_STATUS_5': ['mean'],
        'BALANCE_STATUS_C': ['mean'],
        'BALANCE_STATUS_X': ['mean'],
        'BALANCE_MONTHS_BALANCE_MIN': ['min'],
        'BALANCE_MONTHS_BALANCE_MAX': ['max'],
        'BALANCE_MONTHS_BALANCE_SIZE': ['mean', 'sum']
    })
    bureau_balance.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_balance.columns.tolist()])

    bureau_numerical = bureau.groupby('SK_ID_CURR').agg({
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum']
        })
    bureau_numerical.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_numerical.columns.tolist()])

    bureau_numerical_active = bureau[bureau.CREDIT_ACTIVE == 'Active'].groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum']
    })
    bureau_numerical_active.columns = pd.Index(["ACTIVE_" + e[0] + "_" + e[1].upper() for e in bureau_numerical_active.columns.tolist()])
    
    bureau_numerical_closed = bureau[bureau.CREDIT_ACTIVE == 'Closed'].groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum']
    })
    bureau_numerical_closed.columns = pd.Index(["CLOSED_" + e[0] + "_" + e[1].upper() for e in bureau_numerical_closed.columns.tolist()])
    
    bureau_numerical = bureau_numerical.merge(bureau_numerical_active, left_index=True, right_index=True, how='left').\
                                        merge(bureau_numerical_closed, left_index=True, right_index=True, how='left')

    dfs = []
    for col in bureau.columns:
        if bureau[col].dtype == 'object':
            t = pd.crosstab(bureau.SK_ID_CURR, bureau[col].fillna('NaN'), normalize='index')
            t.columns = pd.Index([t.columns.name + '_' + itm.upper() for itm in t.columns.tolist()])
            dfs.append(t)
    bureau_categorial = pd.concat(dfs, axis = 1)
    
    return pd.concat([bureau_numerical, bureau_categorial, bureau_balance], axis=1).add_prefix('BUREAU_')

b = pd.read_csv(BASEDIR + '/bureau.csv', engine='c')
bb = pd.read_csv(BASEDIR + '/bureau_balance.csv', engine='c')
rst = featurization(b, bb)
train = pd.read_csv(BASEDIR + '/train/application.csv', engine='c')
train[['SK_ID_CURR']].merge(rst, left_on='SK_ID_CURR', right_index=True).to_csv(BASEDIR+'/train/bureau.csv', index=False)
test = pd.read_csv(BASEDIR + '/test/application.csv', engine='c')
test[['SK_ID_CURR']].merge(rst, left_on='SK_ID_CURR', right_index=True).to_csv(BASEDIR+'/test/bureau.csv', index=False)