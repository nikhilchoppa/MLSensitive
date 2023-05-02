import pandas as pd

def macd_pred(csv_file_path):
    # Read in the CSV file
    df = pd.read_csv(csv_file_path)

    # Calculate the MACD
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd
    df['Signal'] = signal

    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change()

    # Make stock predictions based on the MACD
    df['Position'] = 0
    df.loc[signal > macd, 'Position'] = 1
    df.loc[signal < macd, 'Position'] = -1
    df['Strategy'] = df['Position'].shift(1) * df['Returns']
    df['Cumulative_Returns'] = (1 + df['Strategy']).cumprod()

    # Print the cumulative returns
    final_return = df['Cumulative_Returns'].iloc[-1]
    print(f"MACD Cumulative Returns: {final_return:.2f}")

    # Make investment decision based on final return
    if final_return > 1:
        print("This stock has good returns\n")
    else:
        print("This stock has poor returns.\n")


