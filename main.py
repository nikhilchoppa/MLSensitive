import csv
import regression
import macd
import sentimentAnalysisPretrainedBert
import yfinance as yf

def main():

    # Read and parse data
    # TODO: Make this user input
    start_date = '2019-01-01'
    end_date = '2020-12-31'
    stock = "MSFT"
    current_stock = yf.download(stock, start=start_date, end=end_date)
    current_stock["Date"] = current_stock.index
    stock_data = current_stock[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    stock_data.reset_index(drop=True, inplace=True)

    # stock_data.to_csv("data/MSFT.csv", index=False)  # Convert stock data into csv, replaces current one in /data
    # regression_data = "data/MSFT.csv"  # Read the new csv file
    stock_data.to_csv(f"data/{stock}.csv", index=False)
    regression_data = f"data/{stock}.csv"

    # Models
    regression.regression_pred(regression_data)  # Pass the csv file to objects
    macd.macd_pred(regression_data)

if __name__ == "__main__":
    main()