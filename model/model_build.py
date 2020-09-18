###############################
import scipy, os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, preprocessing, metrics   # export_graphviz,
from sklearn.model_selection import train_test_split, learning_curve

import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint

####################################
p19 = pd.read_csv("tai_power_train_1705-1803_utf8.csv",encoding = "utf8")
p19["尖峰負載(MW)"] = p19["尖峰負載(MW)"].astype('float64')
print('p19 :' , p19.shape)

#####################################
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
########################################
p20 = nanfill(p19)
print('p20 :' , p20.shape)
########################################

def get_date(df) :
    df["日期"] = pd.to_datetime(df["日期"], format='%Y%m%d')
    df["month"] = df["日期"].dt.month
    df["date"] = df["日期"].dt.day
    df["day"] = df["日期"].dt.dayofweek
    return df
########################################
p21 = get_date(p20)
print('p21 :' , p21.shape)
########################################

def scale(df) :
    scalar = MinMaxScaler()
    df2 = df.drop(["日期"], axis=1)
    df2 = scalar.fit(df2).transform(df2)
    return df2
########################################
p22 = scale(p21)
print('p22 :' , p22.shape)
########################################

def set_train(train, past , future):
    X_train, Y_train = [], []
    for i in range(len(train)-future-past):
        
        X_train.append( train[i : i + past ] ) 
        
        Y_train.append( train[ i + past : i + past + future , 1 ] )
    
    return np.array(X_train), np.array(Y_train)
###########################################
x_train, y_train = set_train(p22 ,7, 7)
print('x_train :' , x_train.shape)

y_train = y_train[ : , : , np.newaxis]
print('y_train :' , y_train.shape)

############################################
def rnn_model(xshape) :
    # Initialising the RNN
    model = Sequential()
    
    # the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 128, return_sequences = True, input_shape = (xshape[1], xshape[2]) ) )
#    model.add(Lambda(lambda x: x[:, -n:, :]))
    model.add(Dropout(0.1))
    
    # the second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 64, return_sequences = True))
    model.add(Dropout(0.1))
    
    # the output layer output shape: (7, 1)
    model.add(TimeDistributed(Dense(units = 1 )))
#    model.add(RepeatVector(7))
    # Compiling
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.summary()
    return model
###############################################
model = rnn_model(x_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
model.fit(x_train, y_train, epochs=1000, batch_size=64, callbacks=[callback])
###############################################

'''
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

'''
##############################################





















































































































