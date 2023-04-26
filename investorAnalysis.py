import yfinance as yf


def get_recommendations(ticker):
    stock = yf.Ticker(ticker)
    recommendations = stock.recommendations
    return recommendations


ticker = "AAPL"
recommendations = get_recommendations(ticker)
print(recommendations.tail())  # Show the most recent recommendations
