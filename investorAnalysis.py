import pandas as pd
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from yfinance import ticker


def get_recommendations(ticker):
    stock = yf.Ticker(ticker)
    long_name = stock.info['longName']

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0'
    }

    url = f'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=recommendationTrend'
    r = requests.get(url, headers=headers)
    print(r)

    if not r.ok:
        print("Error fetching data.")
        return

    result = r.json()['quoteSummary']['result']

    if not result:
        print(f"No recommendation data available for {long_name} ({ticker}).")
        return

    data = result[0]['recommendationTrend']['trend']

    periods = []
    strong_buys = []
    buys = []
    holds = []
    sells = []
    strong_sells = []

    for trend in data:
        periods.append(trend['period'])
        strong_buys.append(trend['strongBuy'])
        buys.append(trend['buy'])
        holds.append(trend['hold'])
        sells.append(trend['sell'])
        strong_sells.append(trend['strongSell'])

    dataframe = pd.DataFrame({
        'Period': periods,
        'Strong Buy': strong_buys,
        'Buy': buys,
        'Hold': holds,
        'Sell': sells,
        'Strong Sell': strong_sells
    })

    dataframe = dataframe.set_index('Period')
    print(dataframe)

    ax = dataframe.plot.bar(rot=0)
    ax.set_title(f'{long_name} ({ticker}) Recommendation Trends')
    ax.set_ylabel('Recommendation Counts')
    plt.show()

