import warnings
import yfinance as yf
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch.utility.exceptions import ConvergenceWarning
from tabulate import tabulate
from arch import arch_model

warnings.filterwarnings("ignore", "", ConvergenceWarning)


def garch_pred(dataset):
    # ticker_symbol = input("Enter the stock ticker symbol :")
    # tesla_data = yf.Ticker(ticker_symbol)
    tesla_data = yf.Ticker(dataset)

    # history function helps to extract stock information.
    # setting period parameter to max to get information for the maximum amount of time.
    tsla_data = tesla_data.history(period='max')

    # Resetting the index
    tsla_data.reset_index(inplace=True)

    # display the first five rows
    tsla_data.head()
    tsla_data.set_index("Date", inplace=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.spines[['top', 'right']].set_visible(False)
    # plt.plot(tsla_data['Close'], label = 'Daily Returns')
    # plt.legend(loc='upper right')
    # plt.title('Daily Returns Over Time')

    daily_volatility = tsla_data['Close'].std()

    monthly_volatility = math.sqrt(21) * daily_volatility

    annual_volatility = math.sqrt(252) * daily_volatility

    # print(tabulate([['Tesla',daily_volatility,monthly_volatility,annual_volatility]],headers = ['Daily Volatility %', 'Monthly Volatility %',
    #        'Annual Volatility %'],tablefmt = 'fancy_grid',stralign='center',numalign='center',floatfmt=".2f"))

    # BUILDING THE GARCH MODEL

    garch_model = arch_model(tsla_data['Close'], p=1, q=1,
                             mean='constant', vol='GARCH', dist='normal')

    gm_result = garch_model.fit(disp='off')
    # print(gm_result.params)

    # print('\n')

    gm_forecast = gm_result.forecast(horizon=5)
    # print(gm_forecast.variance[-1:])

    # Rolling Predictions

    rolling_predictions = []
    test_size = 365

    for i in range(test_size):
        train = tsla_data['Close'][:-(test_size - i)]
        model = arch_model(train, p=1, q=1)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

    rolling_predictions = pd.Series(rolling_predictions, index=tsla_data['Close'].index[-365:])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.spines[['top', 'right']].set_visible(False)
    # plt.plot(rolling_predictions)
    # plt.title('Rolling Prediction')

    # Rolling Forecast
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.grid(which="major", axis='y', color='#758D99', alpha=0.3, zorder=1)
    ax.spines[['top', 'right']].set_visible(False)
    # plt.plot(tsla_data['Close'][-365:])
    # plt.plot(rolling_predictions)
    # plt.title(f"{dataset} Volatility Prediction - Rolling Forecast")
    # plt.legend(['True Daily Returns', 'Predicted Volatility'])
    # plt.show()

    # Calculate Sharpe ratio
    returns = tsla_data['Close'].pct_change()
    sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())

    if sharpe_ratio < 1:
        print("Based on the GARCH Sharpe ratio, it is not a good time to invest.\n")
        return 0
    else:
        print("Based on the GARCH Sharpe ratio, it is a good time to invest.\n")
        return 1
