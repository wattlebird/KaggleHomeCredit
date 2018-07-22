import pandas as pd
import numpy as np
from setting import *

def process(df):
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    return df[df['CODE_GENDER'] != 'XNA']

train = pd.read_csv(BASEDIR + '/application_train.csv', engine='c')
process(train).to_csv(BASEDIR+'/train/application.csv', index=False)
test = pd.read_csv(BASEDIR + '/application_test.csv', engine='c')
process(test).to_csv(BASEDIR+'/test/application.csv', index=False)
