import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess the data
tesla_data = yf.Ticker('TSLA')

# history function helps to extract stock information.
# setting period parameter to max to get information for the maximum amount of time.
tsla_data = tesla_data.history(period='max')

# Resetting the index
tsla_data.reset_index(inplace=True)

# display the first five rows
tsla_data.head()

#Handling missing values

data = tsla_data
data = data.drop(['Date', 'Dividends', 'Stock Splits'], axis=1)  # drop non-numeric columns
data = data.fillna(data.mean())
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
# Split the data into training and testing sets
train_size = int(len(data_scaled) * 0.7)
test_size = len(data_scaled) - train_size
train_data, test_data = data_scaled[0:train_size,:], data_scaled[train_size:len(data_scaled),:]

# Define the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(train_data.shape[1], 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(train_data, train_data[:, 0], epochs=1, batch_size=1, verbose=2)

# Make predictions using the LSTM model
train_predict_lstm = model_lstm.predict(train_data)
test_predict_lstm = model_lstm.predict(test_data)

# Evaluate the LSTM model
mse_lstm = mean_squared_error(train_data[:, 0], train_predict_lstm[:, 0])
rmse_lstm = np.sqrt(mse_lstm)
print("RMSE of LSTM:", rmse_lstm)

# Define the Random Forest model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(train_data, train_data)

# Make predictions using the Random Forest model
train_predict_rf = model_rf.predict(train_data)
test_predict_rf = model_rf.predict(test_data)

# Evaluate the Random Forest model
mse_rf = mean_squared_error(train_data, train_predict_rf)
rmse_rf = np.sqrt(mse_rf)
print("RMSE of Random Forest:", rmse_rf)

