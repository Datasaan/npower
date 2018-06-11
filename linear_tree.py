# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:54:16 2016

@author: sanjeet
"""

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit as ts
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error as auc
from sklearn.linear_model import LinearRegression as lr

########################################################

def agg_feats(train):
    train_m=train.groupby(['Date'],as_index=False).mean()
    train=pd.merge(train,train_m[['Date','Demand']],on='Date')
    train['Demand_z']=train['Demand_x']-train['Demand_y']
    return train,train_m
def datasets(train):
    to_train_i=train[train.Date.dt.year.isin([2012,2013])]
    to_train_cv=train[train.Date.dt.year==2014][train.Date.dt.month.isin([1,2,3])]
    to_test=train[train.Date.dt.year==2014][train.Date.dt.month.isin([4,5,6,7,8,9])]
    to_train=pd.concat([to_train_i,to_train_cv])
    return to_train,to_test
#######################################################

print 'Data Prep...'
############################################################################
main=pd.read_excel('round_2.xlsx')
main=main.drop(main.columns[2],axis=1)
train=main[main.Demand!='?? FC2 ??']
ref_date=train.Date.min()
test=main[main.Demand=='?? FC2 ??']
train.Demand=train.Demand.astype('float')
############################################################################
train,train_m=agg_feats(train)
fc1=pd.read_csv('forecast1.csv')
fc1,fc1_m=agg_feats(fc1)
fc1_real=train[train.Date>=fc1.Date.min()][train.Date<=fc1.Date.max()]
fc1_real=fc1_real.reset_index(drop=True)
test.drop(labels=['Demand'],axis=1)
test_m=test.groupby(['Date'],as_index=False).mean()
##############################################################################

print 'linear model'
####################################################################
train_m=train_m.fillna(method='ffill')
to_train,to_test=datasets(train_m)
def input(train):
    m=pd.DataFrame()
    #m['year']=train.Date.dt.year
    #m['month']=train.Date.dt.month
    #m['day']=train.Date.dt.day
    #m['weekday']=train.Date.dt.dayofweek
    m['day_count']=(train.Date-ref_date).dt.days
    m['Effective Temperature']=train_m['Effective Temperature']
    #m['Demand']=train['Demand']
    return m
def output(train):
    return (train.Demand)
model1=lr()
model1.fit(input(to_train),output(to_train))
testp_m=pd.DataFrame()
trainp_m=pd.DataFrame()
testp_m['data']=to_test.Demand
trainp_m['data']=to_train.Demand
testp_m['prediction_lm']=model1.predict(input(to_test))
trainp_m['prediction_lm']=model1.predict(input(to_train))
#train['prediction_lm']=model1.predict(input(train))
lin_er=np.sqrt(auc(testp_m.data,testp_m.prediction_lm))
print 'linear model done'
print 'Current error'
print lin_er

test_m['prediction_lm']=model1.predict(input(test_m))
test=pd.merge(test,test_m[['Date','prediction_lm']],on='Date')


print('XGBoost Model for residuals')
###############################################################
to_train,to_test=datasets(train)

def input(train):
    m=pd.DataFrame()
    m['year']=train.Date.dt.year
    m['month']=train.Date.dt.month
    m['day']=train.Date.dt.day
    m['weekday']=train.Date.dt.dayofweek
    m['Period']=train.Period
    m[[u'Temperature', u'Effective Temperature',u'Wind Speed', u'Wind Direction', u'Precipitation Amount',u'Precipitation Type', u'Solar Radiation', u'Humidity', u'Cloud Cover']]=train[[u'Temperature', u'Effective Temperature',u'Wind Speed', u'Wind Direction', u'Precipitation Amount',u'Precipitation Type', u'Solar Radiation', u'Humidity', u'Cloud Cover']]
    return m
def output(train):
    return train.Demand_z
dtrain_full=xgb.DMatrix(data=input(train),label=output(train),missing=float('nan'))
dtrain=xgb.DMatrix(data=input(to_train),label=output(to_train),missing=float('nan'))
dtest=xgb.DMatrix(data=input(to_test),label=output(to_test),missing=float('nan'))
params = {}
params['booster'] = 'gbtree'
params['objective'] ='reg:linear'
params['eval_metric'] = 'rmse'
params['eta'] = 0.05
params['gamma'] = 0.5
params['min_child_weight'] = 4
params['colsample_bytree'] = 0.6
params['subsample'] = 0.99
params['max_depth'] = 7
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 1001
earlystopping=100
watchlist=[(dtrain,'train'),(dtest,'validate')]
model=xgb.train(params,dtrain=dtrain,num_boost_round=1000,evals=watchlist)
n_trees=model.best_ntree_limit
testp=pd.DataFrame()
testp['data']=to_test.Demand_z
testp['prediction_xgb']=model.predict(dtest,ntree_limit=n_trees)
trainp=pd.DataFrame()
trainp['data']=to_train.Demand_z
trainp['prediction_xgb']=model.predict(dtrain,ntree_limit=n_trees)
xgb_er=np.sqrt(auc(testp.data,testp.prediction_xgb))
dtrain_full=xgb.DMatrix(data=input(train),label=output(train),missing=float('nan'))
train['prediction_xgb']=model.predict(dtrain_full,ntree_limit=n_trees)

######################

watchlist=[(dtrain_full,'train'),(dtest,'validate')]
fin_model=xgb.train(params,dtrain=dtrain,num_boost_round=1000,evals=watchlist)
n_trees=fin_model.best_ntree_limit
dtest_full=xgb.DMatrix(data=input(test),missing=float('nan'))
test['prediction_xgb']=fin_model.predict(dtest_full,ntree_limit=n_trees)



#####################
#########################################################################################

#linear model for daily average
#######################################################################################
to_train,to_test=datasets(train)

def input(train):
    m=pd.DataFrame()
    m['prediction_lm']=train['prediction_lm']
    m['prediction_xgb']=train['prediction_xgb']
    return m
def output(train):
    return (train.Demand_x)
model2=lr()
model2.fit(input(to_train),output(to_train))
testp['data']=to_test.Demand_x
trainp['data']=to_train.Demand_x
testp['prediction']=model2.predict(input(to_test))
trainp['prediction']=model2.predict(input(to_train))
train['prediction']=model2.predict(input(train))
stc_er=np.sqrt(auc(testp.data,testp.prediction))
######################################################################


## linear model on output of xgboost and linear daily demand model


fin_model2=lr()
fin_model2.fit(input(train),output(train))
test['Demand']=fin_model2.predict(input(test))


######################################################################

print 'linear model done'
print 'Current error'
print stc_er
