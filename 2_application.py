import pandas as pd
import numpy as np
from setting import *

def feature(df):
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    return df[['SK_ID_CURR', 'DAYS_EMPLOYED_PERC', 'INCOME_CREDIT_PERC', 'INCOME_PER_PERSON', 'ANNUITY_INCOME_PERC', 'PAYMENT_RATE']]

train = pd.read_csv(BASEDIR + '/train/application.csv', engine='c')
feature(train).to_csv(BASEDIR+'/train/application_ext.csv', index=False)
test = pd.read_csv(BASEDIR + '/test/application.csv', engine='c')
feature(test).to_csv(BASEDIR+'/test/application_ext.csv', index=False)
