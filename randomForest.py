import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def random_forest_pred(dataset):
    # Load and preprocess the data
    stk_data = yf.Ticker(dataset)

    # Extract stock information.
    stock_data = stk_data.history(period='max')

    # Resetting the index
    stock_data.reset_index(inplace=True)

    #Handling missing values
    data = stock_data
    data = data.drop(['Date', 'Dividends', 'Stock Splits'], axis=1)  # drop non-numeric columns
    data = data.fillna(data.mean())
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Calculate the daily return
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    data_scaled['Return'] = data_scaled['Close'].pct_change()
    data_scaled.dropna(inplace=True)

    # Classify the return as '1' if positive and '0' if not
    data_scaled['Return'] = (data_scaled['Return'] > 0).astype(int)

    # Split the data into training and testing sets
    train_size = int(len(data_scaled) * 0.7)
    test_size = len(data_scaled) - train_size
    train_data, test_data = data_scaled.iloc[0:train_size,:], data_scaled.iloc[train_size:len(data_scaled),:]

    # 'Return' as target
    train_y = train_data['Return']
    test_y = test_data['Return']

    # Rest of the data as input
    train_X = train_data.drop('Return', axis=1)
    test_X = test_data.drop('Return', axis=1)

    # Define the Random Forest model
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(train_X, train_y)

    # Make predictions using the Random Forest model
    train_predict_rf = model_rf.predict(train_X)
    test_predict_rf = model_rf.predict(test_X)

    # Evaluate the Random Forest model
    accuracy_rf = accuracy_score(test_y, test_predict_rf)
    print("Accuracy of Random Forest:", accuracy_rf)

    # Classify the stock as either 'good' or 'poor'
    if accuracy_rf > 0.65:
        print("Random Forest Accuracy: Performing Well.\n")
        return 1
    else:
        print("Random Forest Accuracy: Performing Poorly.\n")
        return 0
