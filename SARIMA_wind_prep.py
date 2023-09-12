import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import warnings
from statsmodels.tsa.arima.model import ARIMA


day = 60*60*24
year = 365.2425*day
#ADDING THE PATH OF THE SAVED DATA 
model_path:str=r"C:\Users\milto\Desktop\Github\Tensorflow\Wind_Prediction"
filepath :str= r"C:\Users\milto\Desktop\batteryData\GenPriceData.xlsx"
wind_col = "Generation in MWh"

#CALLING THE THE DATA FROM THE EXCEL AND USING THE DATATIME AS INDEX
df=pd.read_excel(filepath, index_col= "Datetime", parse_dates=True)
df = df.dropna() #SEARCH FOR MISSING DATA

#df=df.iloc[:,:1]
#print(df)
#plt.plot(df[wind_col])
#plt.show()

#PMDARIMA FACES SOME HARMLESS ISSUES
#SO WE NEED TO USE FILTERWARING TO IGNORE HARMLESS WARNINGS
warnings.filterwarnings("ignore")

train_model = df.iloc[:-30]
test_model = df.iloc[-30:]


####CHECKING FOR MODELS STAATIONARITY
#adf_test = adfuller(train_model[wind_col])
#print(f'p-value: {adf_test[1]}')
##If p-value < 0.05 : stationary
##else: not stationary

#Auto search for the best order (p,d,q)
model = auto_arima(train_model[wind_col],
                    trace=True, suppress_warnings=True)

model.summary()


predictions = pd.Series(model.predict(n_periods =len(test_model)))
predictions.index = test_model.index
print(predictions[:5])
plt.plot(predictions)
plt.plot(test_model)
plt.show()







