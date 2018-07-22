import pandas as pd
import numpy as np
from setting import *

application = pd.read_csv(BASEDIR + '/train/application.csv', engine='c')
application_ext = pd.read_csv(BASEDIR + '/train/application_ext.csv', engine='c')
bureau = pd.read_csv(BASEDIR + '/train/bureau.csv', engine='c')
previous = pd.read_csv(BASEDIR + '/train/previous.csv', engine='c')
installment = pd.read_csv(BASEDIR + '/train/installment.csv', engine='c')
pos = pd.read_csv(BASEDIR + '/train/pos.csv', engine='c')
creditcard = pd.read_csv(BASEDIR + '/train/creditcard.csv', engine='c')
train = application.merge(application_ext, on='SK_ID_CURR', how='left')\
                   .merge(bureau, on='SK_ID_CURR', how='left')\
                   .merge(previous, on='SK_ID_CURR', how='left')\
                   .merge(installment, on='SK_ID_CURR', how='left')\
                   .merge(pos, on='SK_ID_CURR', how='left')\
                   .merge(creditcard, on='SK_ID_CURR', how='left')
train.to_csv(BASEDIR + "/train/train.csv", index=False)

application = pd.read_csv(BASEDIR + '/test/application.csv', engine='c')
application_ext = pd.read_csv(BASEDIR + '/test/application_ext.csv', engine='c')
bureau = pd.read_csv(BASEDIR + '/test/bureau.csv', engine='c')
previous = pd.read_csv(BASEDIR + '/test/previous.csv', engine='c')
installment = pd.read_csv(BASEDIR + '/test/installment.csv', engine='c')
pos = pd.read_csv(BASEDIR + '/test/pos.csv', engine='c')
creditcard = pd.read_csv(BASEDIR + '/test/creditcard.csv', engine='c')
test = application.merge(application_ext, on='SK_ID_CURR', how='left')\
                   .merge(bureau, on='SK_ID_CURR', how='left')\
                   .merge(previous, on='SK_ID_CURR', how='left')\
                   .merge(installment, on='SK_ID_CURR', how='left')\
                   .merge(pos, on='SK_ID_CURR', how='left')\
                   .merge(creditcard, on='SK_ID_CURR', how='left')
test.to_csv(BASEDIR + "/test/test.csv", index=False)