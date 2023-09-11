# Machine_learning
LSTM_Forecasting
This project focuses on forecasting wind generation in megawatt-hours (MWh) using Long Short-Term Memory (LSTM) model. 
The goal is to provide accurate wind generation forecasts based on historical data.

Clone this repository to your local machine:   
    git clone https://github.com/Miltos99/Machine_learning/wind-generation-forecast.git

Navigate to the project directory:
    cd wind-generation-forecast


Data Preparation
The wind generation data is loaded from an Excel file (GenPriceData.xlsx).
Cyclic patterns related to time, such as daily and yearly seasonality, are encoded to improve modeling.
Data is split into training and validation sets.

Model Training
The code includes an LSTM-based neural network model for time series forecasting.
The model is trained on the training dataset using Mean Squared Error (MSE) as the loss function and RMSprop as the optimizer.

Predictions
The trained model is used to make future wind generation predictions.
The predicted values are stored in a DataFrame and plotted against the actual data for visualization.

Results
The project's results, including forecasted wind generation values, are available in the results directory.
Evaluation metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) are computed and included in the results.
