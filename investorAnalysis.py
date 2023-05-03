import pandas as pd
import yfinance as yf
import numpy as  np
import pandas as p
import requests




def get_recommendations(ticker):
    stock = yf.Ticker(ticker)
    recommendations = []
    for tick in ticker:
        pre_url = 'https://query2.finance.yahoo.com/v10/finance/quoteSu'
        post_url = ''
        url = pre_url + post_url
        r= requests.get(url)
        if not r.ok:
            recommendation = 6
        try:
            result = r.json()['quoteSummary']['result'][0]
            recommendation = result['financialData']['recommendationMean']['fmt']
        except:
            recommendation=6

        recommendations.append(recommendation)
        print('{} has an average recommendation of :'.format(tick),recommendation)

        dataframe = pd.DataFrame(list(zip(tick,recommendations)),columns=['Company','Recommendations'])
        print(dataframe)
        dataframe = dataframe.set_index('Company')
        dataframe.to.csv('RecommendationTable.csv')
        dataframe.sort_values(['Recommendations']).head(5)
        dataframe['Recommendations'] = dataframe['Recommendations'].astype(float)

    return recommendations


ticker = "TSLA"
recommendations = get_recommendations(ticker)
print(recommendations)  # Show the most recent recommendations
