###############
import scipy, os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split, learning_curve

import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json

########################

#############
xtest = pd.read_csv("tai_power_test_1805-1902_utf8.csv",encoding = "utf8")
xtest["尖峰負載(MW)"] = xtest["尖峰負載(MW)"].astype('float64')
print('xtest :' , xtest.shape)
#############################

def nanfill(df) :
    print('data variables with NAN :')
    i = 0
    while i < len(df.columns) :
        if df[ df.columns[i] ].isnull().values.any() == True :
            print(df.columns[i] , ':', end='\t')
            print(df[ df.columns[i] ].isnull().values.sum() )
            df[ df.columns[i] ] = df[ df.columns[i] ].fillna(0)
            print('-'*15)
        i += 1
    print('-'*30)
    i = 0
    while i < len(df.columns) :
        if df[ df.columns[i] ].isnull().values.any() == True :
            print(df.columns[i] , ':', end='\t')
            print(df[ df.columns[i] ].isnull().values.sum() )
            print('-'*15)
        i += 1
    print('-'*30)
    return df
############################
xtest = nanfill(xtest)
print('xtest :' , xtest.shape)
#############################

def get_date(df) :
    df["日期"] = pd.to_datetime(df["日期"], format='%Y%m%d')
    df["month"] = df["日期"].dt.month
    df["date"] = df["日期"].dt.day
    df["day"] = df["日期"].dt.dayofweek
    return df

#################################

xtest2 = get_date(xtest)
print('xtest2 :' , xtest2.shape)


##################################
scalar2 = MinMaxScaler()
x_test = xtest2.drop(["日期"], axis=1)
x_test = scalar2.fit(x_test).transform(x_test)
print('x_test :' , x_test.shape)

##################################
y_test = []
for i in range(len(x_test)-7-7):
    y_test.append( x_test[i : i + 7 ] ) 

y_test = np.array(y_test)
print('y_test :' , y_test.shape)
##################################

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("loaded_model done")

###################################

pred_peak = loaded_model.predict(y_test)
print('pred_peak :' , pred_peak.shape)

pred_peak_fianl = pred_peak[289]
print(pred_peak_fianl)

scalar3 = MinMaxScaler()
real_peak = xtest2['尖峰負載(MW)'].values
real_peak = real_peak[ : , np.newaxis]
real_peak2 = scalar3.fit(real_peak).transform(real_peak)
pred_peak_mw = scalar3.inverse_transform(pred_peak_fianl)
print(pred_peak_mw)


peak_load = pred_peak_mw.reshape(7)
print('peak_load :' , peak_load)
peak_load = pd.DataFrame(peak_load)
peak_load.columns = ['peak_load(MW)']
print('peak_load :' , peak_load.shape)
print(pred_peak_mw.shape)

#####################################
pred0 = pd.read_csv('tai_power_predict_1904_utf8.csv' , encoding = "utf8")
pred = pd.concat((pred0 , peak_load) , axis=1)
print('pred :' , pred.shape)
pred.to_csv("submission.csv", index = False, header = True)




























