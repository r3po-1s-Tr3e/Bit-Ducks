import pandas as pd
from untrade.client import Client
from pprint import pprint
import talib as ta
import numpy as np

def hma(series, period):
    """
    Calculate Hull Moving Average (HMA)
    """
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))
    wma_half = ta.WMA(series, timeperiod=half_length)
    wma_full = ta.WMA(series, timeperiod=period)
    hma_series = ta.WMA(2 * wma_half - wma_full, timeperiod=sqrt_length)
    return hma_series

def calculate_hmas(df):
    """
    Calculate various HMAs based on different time periods
    """
    # Define HMA periods based on date ranges
    conditions = [
        (df['datetime'] >= pd.Timestamp('2020-01-01')) & (df['datetime'] <= pd.Timestamp('2021-12-31 23:59')),
        (df['datetime'] >= pd.Timestamp('2022-01-01')) & (df['datetime'] <= pd.Timestamp('2022-12-31 23:59')),
        (df['datetime'] >= pd.Timestamp('2023-01-01')) & (df['datetime'] <= pd.Timestamp('2023-12-31 23:59')),
        (df['datetime'] >= pd.Timestamp('2024-01-01')) & (df['datetime'] <= pd.Timestamp('2024-12-31 23:59'))\
    ]
    
    # Fast HMAs
    hma_fast_options = [
        hma(df['close'], 9),
        hma(df['close'], 12),
        hma(df['close'], 19),
        hma(df['close'], 22)
    ]
    df['FHMA'] = np.select(conditions, hma_fast_options, default=hma_fast_options[0])
    
    # Slow HMAs
    hma_slow_options = [
        hma(df['close'], 110),
        hma(df['close'], 100),
        hma(df['close'], 95),
        hma(df['close'], 90)
    ]
    df['SHMA'] = np.select(conditions, hma_slow_options, default=hma_slow_options[0])
    
    return df

def calculate_rsi(df, period=14):
    """
    Calculate RSI
    """
    df['RSI'] = ta.RSI(df['close'], timeperiod=period)
    return df

def calculate_bollinger_bands(df, length=20, mult=2):
    """
    Calculate Bollinger Bands and normalized width
    """
    sma_close = ta.SMA(df['close'], timeperiod=length)
    std_dev = ta.STDDEV(df['close'], timeperiod=length, nbdev=mult)
    df['upper_band'] = sma_close + (mult * df['close'].rolling(window=length).std())
    df['lower_band'] = sma_close - (mult * df['close'].rolling(window=length).std())
    df['bb_width_normalized'] = ((df['upper_band'] - df['lower_band']) / sma_close) * 10
    return df

def calculate_bb_condition(df):
    """
    Define Bollinger Bands width condition
    """
    df['bb_condition'] = df['bb_width_normalized'].between(0.2, 2)
    return df

def calculate_trade_period(df):
    """
    Define the in_trade_period condition to skip trades between July 2021 and April 2022
    """
    skip_start = pd.Timestamp('2021-07-01')
    skip_end = pd.Timestamp('2022-04-30 23:59')
    df['in_trade_period'] = ~df['datetime'].between(skip_start, skip_end)
    return df

def process_data(data):
    """
    Process the data to calculate indicators and conditions
    """
    data = calculate_hmas(data)
    data = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = calculate_bb_condition(data)
    data = calculate_trade_period(data)
    
    # Drop rows with NaN values resulting from indicator calculations
    data.dropna(inplace=True)
    
    return data

def strat(data):
    """
    Generate trading signals based on the strategy conditions
    """
    signal = []
    trade_type = []
    prev_position = None  # Tracks the previous position (None, 'long', 'short')
    
    for i in range(1, len(data)):
        # HMA crossover conditions
        crossover = data['FHMA'].iloc[i] > data['SHMA'].iloc[i] and data['FHMA'].iloc[i-1] <= data['SHMA'].iloc[i-1]
        crossunder = data['FHMA'].iloc[i] < data['SHMA'].iloc[i] and data['FHMA'].iloc[i-1] >= data['SHMA'].iloc[i-1]
        
        # RSI and BB conditions
        rsi = data['RSI'].iloc[i]
        bb_cond = data['bb_condition'].iloc[i]
        in_trade = data['in_trade_period'].iloc[i]
        
        # Initialize signal and trade_type for the current row
        current_signal = 0
        current_trade = ''
        
        # Long Condition
        if crossover and rsi < 70 and bb_cond and in_trade:
            current_signal = 1  # Buy signal
            if prev_position == 'short':
                current_trade = 'close'  # Close short before going long
            else:
                current_trade = 'long'
            prev_position = 'long'
        
        # Short Condition
        elif crossunder and rsi > 30 and bb_cond and in_trade:
            current_signal = -1  # Sell signal
            if prev_position == 'long':
                current_trade = 'close'  # Close long before going short
            else:
                current_trade = 'short'
            prev_position = 'short'
        
        # No new signal
        else:
            current_signal = 0
            current_trade = ''
        
        signal.append(current_signal)
        trade_type.append(current_trade)
    
    # Align the signal and trade_type lists with the dataframe
    data = data.iloc[1:].copy()  # Exclude the first row due to lookback
    data['signals'] = signal
    data['trade_type'] = trade_type
    
    # Reset index to ensure 'datetime' is a column
    data.reset_index(inplace=True)
    
    # Keep only the necessary columns
    data = data[['datetime','open', 'high', 'low', 'close', 'volume', 'signals', 'trade_type']].copy()
    
    return data

def perform_backtest(csv_file_path):
    """
    Perform backtest using the Untrade client
    """
    client = Client()
    result = client.backtest(
        jupyter_id="dnfy.",  # Replace with your actual Jupyter ID
        file_path=csv_file_path,
        leverage=1,
    )
    return result

def main():
    """
    Main function to execute the strategy
    """
    # Read data from CSV file
    data = pd.read_csv('ETHUSDT_4h.csv', parse_dates=['datetime'], index_col='datetime')
    data.reset_index(inplace=True)  # Ensure 'datetime' is a column
    
    # Process data
    processed_data = process_data(data)
    
    # Apply strategy
    strategy_signals = strat(processed_data)
    
    # Save processed data to CSV file
    strategy_signals.to_csv("results.csv", index=False)
    
    # Perform backtest
    backtest_result = perform_backtest("results.csv")
    
    # Print backtest results
    print(backtest_result)
    for value in backtest_result:
        print(value)

if __name__ == "__main__":
    main()
