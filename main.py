import csv
import regression
import macd
import sentimentAnalysisPretrainedBert
import yfinance as yf

def main():

    running = True
    while running:
        # User input
        stock = input("Please enter in a stock to search: ")
        start_date = input("Please enter in a start date (YYYY-MM-DD): ")
        end_date = input("Please enter in an end date (YYYY-MM-DD): ")

        # Read and parse the stock data
        current_stock = yf.download(stock, start=start_date, end=end_date)
        current_stock["Date"] = current_stock.index
        stock_data = current_stock[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        stock_data.reset_index(drop=True, inplace=True)

        # Put data into csv files located at data/filename.csv, for regression and macd models
        stock_data.to_csv(f"data/{stock}.csv", index=False)
        regression_data = f"data/{stock}.csv"

        # Models
        regression.regression_pred(regression_data)  # Pass the csv file to objects
        macd.macd_pred(regression_data)


if __name__ == "__main__":
    main()