import csv
import regression
import macd
import sentimentAnalysisPretrainedBert
import yfinance as yf

def main():

    # Please create a user input loop that asks users to enter in stock data with start and end dates
    running = True
    while running:
        # Ask the user to enter in a stock ticker
        stock = input("Please enter in a stock to search: ")
        # Ask the user to enter in a start date
        start_date = input("Please enter in a start date (YYYY-MM-DD): ")
        # Ask the user to enter in an end date
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