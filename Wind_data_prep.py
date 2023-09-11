import pandas as pd 
import numpy as np 
import tensorflow as tf
from tensorflow import _keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from statsmodels.tsa.arima_model import ARIMA
#from statsmodels.tsa.arima_model import
import statsmodels.api as sm
day = 60*60*24
year = 365.2425*day
#ADDING THE PATH OF THE SAVED DATA 
model_path:str=r"C:\Users\milto\Desktop\Github\Tensorflow\Wind_Prediction"
filepath :str= r"C:\Users\milto\Desktop\batteryData\GenPriceData.xlsx"


#CREATING AN INPUT ARRAY WITH SIZE:WINDOW_SIZE
#AND OUTPUT ARRAY WITH SIZE ONE 
def df_to_X_Y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]] 
    X.append(row) # Input: X = [[,...,], [,...,] ...]
    label = df_as_np[i+window_size][0]
    y.append(label) #Output: Y = [[],[],... ]
  return np.array(X), np.array(y)

#THIS FUNCTION HELP THE MODEL TO CONVERGE FASTER
#AND MAKE THEM LESS SENSITIVE TO THE SCALE OF INPUT FEATURES

def Standardization(X):
  X[:, :, 0] = (X[:, :, 0] - wind_training_mean) / wind_training_std
  return X

"""
def Min_Max(X):
  min_value = np.min(X[:,:,0], axis = 0)
  max_value = np.max(X[:,:,0],axis = 0)
  X[:,:,0] = (X[:,:,0] - min_value) / (min_value - max_value)
  return X
"""

#THIS FUNCTION IS USED TO ENCODE CYCLIC PATTERNS IN TIME SERIES DATA
def data_structure(wind_df:pd.DataFrame):
  wind_df.index=wind_df.pop("Datetime")
  wind_df['Seconds'] = wind_df.index.map(pd.Timestamp.timestamp)
  #INTODUCING NEW TIMESERIES RELATIONSHIPS
  wind_df['Day sin'] = np.sin(wind_df['Seconds'] * (2* np.pi / day))
  wind_df['Day cos'] = np.cos(wind_df['Seconds'] * (2 * np.pi / day))
  wind_df['Year sin'] = np.sin(wind_df['Seconds'] * (2 * np.pi / year))
  wind_df['Year cos'] = np.cos(wind_df['Seconds'] * (2 * np.pi / year))
  return wind_df.drop('Seconds', axis=1)


#CALLING THE DATA FROM THE FILEPATH AND CREATING THE FORMAT WE WANT 
df=pd.read_excel(filepath)
df=df.iloc[:,0:2]

#df['Generation in MWh'].plot(figsize=(12,5))
#plt.show()



# Make predictions with the fitted model


X2, Y2 = df_to_X_Y(data_structure(df)) # X2: ARRAY = [[,...,],...], Y2:ARRAY = [[]]

#SEPARATING THE DATA INTO TRAIN , VALIDATE AND TEST SLICES
X2_train, Y2_train = X2[:39000], Y2[:39000]
X2_val, Y2_val = X2[39000:], Y2[39000:]
#X2_test, y2_test = X2[40000:], y2[40000:]


wind_training_mean = np.mean(X2_train[:, :, 0])
wind_training_std = np.std(X2_train[:, :, 0])
                           

X2_train = Standardization(X2_train)
X2_val = Standardization(X2_val)
#Standardization(X2_test)

"""
X2_train = Min_Max(X2_train)
X2_val = Min_Max(X2_val)
"""


#CREATING THE NN-MODEL
#SETTING THE  MODEL PARAMETERS
model = Sequential()
model.add(InputLayer((5, 5)))
model.add(LSTM(64))
model.add(Dense(8, 'elu'))
model.add(Dense(1, activation='linear'))
model.summary()


cp4 = ModelCheckpoint('model/', save_best_only=True)
model.compile(loss=MeanSquaredError(), optimizer=RMSprop(learning_rate=0.01), metrics=[RootMeanSquaredError()])
model.fit(X2_train, Y2_train, validation_data=(X2_val, Y2_val), epochs=40, batch_size=128)


#SAVING THE MODEL AFTER BEING TRAINED
model.save(model_path + str(r"\wind_model.h5"))