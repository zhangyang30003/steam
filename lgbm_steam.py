# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:54:44 2019

@author: Administrator
"""

import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm.sklearn import LGBMRegressor
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
 
 
train = pd.read_csv(r"C:\Users\Administrator\Desktop\AI_learning\steam\zhengqi_train.csv")
test = pd.read_csv(r"C:\Users\Administrator\Desktop\AI_learning\steam\zhengqi_test.csv")

y_train = train[['target']]
x_train = train.drop('target', axis = 1)
x_test = test

#去除不规则项
all_data = pd.concat([train, test], axis=0, ignore_index = False)
#all_data.drop(["V5","V9","V11","V17","V22","V28"],axis=1,inplace=True)

#normalise
cols_numeric=list(all_data.columns)
def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())
scale_cols = [col for col in cols_numeric if col!='target']
all_data[scale_cols] = all_data[scale_cols].apply(scale_minmax,axis=0)
all_data = all_data.drop("target", axis = 1)

#all_train_data
x_train = all_data[:2888]
x_test = all_data[2888:]
#val and train
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle=True, random_state=42)

########################################################################
print('start ML')
score =[]
train_range = range(1,1000,10)
for i in train_range: 
    print(i)
    lgr = LGBMRegressor(    
        learning_rate=0.05,
        n_estimators=i,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=2019
        )
    lgr.fit(x_train,y_train)
    mse = mean_squared_error(y_val, lgr.predict(x_val))
#    print(mse)
    score.append(mse)

plt.plot(train_range, score)

result = pd.DataFrame(lgr.predict(x_test))
result.to_csv('sub_8-6.txt', index = False, header = 0)
