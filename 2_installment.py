import pandas as pd
import numpy as np
import re
from setting import *


def featurization(df):
    # Percentage and difference paid in each dftallment (amount paid and dftallment value)
    df['PAYMENT_PERC'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
    df['PAYMENT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    df['DPD'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['DBD'] = df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']
    df['DPD'] = df['DPD'].apply(lambda x: x if x > 0 else 0)
    df['DBD'] = df['DBD'].apply(lambda x: x if x > 0 else 0)

    df_numerical = df.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
        'SK_ID_CURR': ['size']
    })
    df_numerical.columns = pd.Index([re.sub('[^0-9A-Z_]+', '_', (e[0] + "_" + e[1]).upper()) for e in df_numerical.columns.tolist()])

    return df_numerical.add_prefix('INS_')

pos = pd.read_csv(BASEDIR + '/installments_payments.csv', engine='c')
rst = featurization(pos)
train = pd.read_csv(BASEDIR + '/train/application.csv', engine='c')
train[['SK_ID_CURR']].merge(rst, left_on='SK_ID_CURR', right_index=True).to_csv(BASEDIR+'/train/installment.csv', index=False)
test = pd.read_csv(BASEDIR + '/test/application.csv', engine='c')
test[['SK_ID_CURR']].merge(rst, left_on='SK_ID_CURR', right_index=True).to_csv(BASEDIR+'/test/installment.csv', index=False)