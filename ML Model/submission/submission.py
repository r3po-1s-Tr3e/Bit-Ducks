# In[1]: import of required libs
print("in[1] started")
# !pip install lightgbm==4.2.0 -i https://mirrors.aliyun.com/pypi/simple/
# !pip install catboost==1.2.7 -i https://mirrors.aliyun.com/pypi/simple/
# !pip install xgboost==2.0.3 -i https://mirrors.aliyun.com/pypi/simple/
# !pip install joblib==1.4.2 -i https://mirrors.aliyun.com/pypi/simple/
# !pip install polars==0.18.3 -i https://mirrors.aliyun.com/pypi/simple/

import os
import joblib

import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

print("import of required libs done")


# In[5]: setting minor variables
# If the local directory exists, use it; otherwise, use the Kaggle input directory

# Number of validation dates to use


# Number of dates to skip from the beginning of the dataset
skip_dates = 2

# Number of folds for cross-validation
N_fold = 5

# In[2]:If in training mode, load the training data and test data
stock = "BTC"
interval = "1min"
TRAINING = True
num_valid_dates = 90
global_input_path_train="... set the input path of the file that has train data"
global_input_path_test="... set the input path of the file that has test data"
test_data_points=10000
if TRAINING and stock == "ETH" and interval == "1min":
    # loading the data from the folder and makinng a single  df-
    input_path = "eth-1-min-data" if os.path.exists("eth-1-min-data") else "eth-1-min-data"


    # using 3 years data for training -

    samples = []
    for i in range(3):
        file_path = f"{input_path}/202{i}_2{i+1}_1min_eth.csv"
        chunk = pd.read_csv(file_path)
        sample_chunk = chunk.sample(
            n=100000 * 5, random_state=42
        )  # For example, 100 rows
        samples.append(sample_chunk)

    ## test data set - 1 year as training-

    file_path = f"{input_path}/2023_24_1min_eth.csv"
    test = pd.read_csv(file_path)
    unprocessed_test = test.copy()

    # Convert 'datetime' to datetime format
    test["datetimes"] = pd.to_datetime(test["datetime"], errors="coerce")

    # Check for any invalid values in the datetime column
    if test["datetimes"].isnull().any():
        print("Warning: Some entries in the 'datetime' column could not be parsed!")
        print(
            test[test["datetimes"].isnull()]
        )  # Display rows with invalid datetime values

    # Proceed with further processing
    test["date"] = test["datetimes"].dt.date  # Extract the date part
    test["date_id"] = (
        test["date"].astype("category").cat.codes
    )  # Create unique IDs for each date

    # print(test.head(100))

    df = pd.concat(samples, ignore_index=True)
    df["datetimes"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Check for any invalid values in the datetime column
    if df["datetimes"].isnull().any():
        print("Warning: Some entries in the 'datetime' column could not be parsed!")
        print(
            df[test["datetimes"].isnull()]
        )  # Display rows with invalid datetime values

    # Proceed with further processing
    df["date"] = df["datetimes"].dt.date  # Extract the date part
    df["date_id"] = (
        df["date"].astype("category").cat.codes
    )  # Create unique IDs for each date

    print(" all files imported")

    dates = df["datetime"].unique()
    valid_dates = dates[-num_valid_dates:]

    # Define training dates as all dates except the last `num_valid_dates` dates
    train_dates = dates[:-num_valid_dates]


elif TRAINING and stock == "BTC" and interval == "1min":
    input_path = "btc-1-min-data" if os.path.exists("btc-1-min-data") else "btc-1-min-data"


    # loading the data from the folder and makinng a single  df-

    # using 3 years data for training -
    
    file_path = f"{input_path}/BTC_2019_2023_1m.csv"
    chunk = pd.read_csv(file_path)
    # For example, 100 rows

    ## test data set - as last 1 lakh data points and rest above as training-
    test = chunk.copy().iloc[len(chunk) - 100000 :]

    unprocessed_test = test.copy()
    # Convert 'datetime' to datetime format
    test["datetimes"] = pd.to_datetime(test["datetime"], errors="coerce")

    # Check for any invalid values in the datetime column
    if test["datetimes"].isnull().any():
        print("Warning: Some entries in the 'datetime' column could not be parsed!")
        print(
            test[test["datetimes"].isnull()]
        )  # Display rows with invalid datetime values

    # Proceed with further processing
    test["date"] = test["datetimes"].dt.date  # Extract the date part
    test["date_id"] = (
        test["date"].astype("category").cat.codes
    )  
    # Create unique IDs for each date

    print(" all files imported")
    df = chunk.copy().iloc[: len(chunk) - 100000]
    
    df["datetimes"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Check for any invalid values in the datetime column
    if df["datetimes"].isnull().any():
        print("Warning: Some entries in the 'datetime' column could not be parsed!")
        print(
            df[test["datetimes"].isnull()]
        )  # Display rows with invalid datetime values

    # Proceed with further processing
    
    df["date"] = df["datetimes"].dt.date  # Extract the date part
    df["date_id"] = (
        df["date"].astype("category").cat.codes
    )  # Create unique IDs for each date

    dates = df["datetime"].unique()
    valid_dates = dates[-num_valid_dates:]

    # Define training dates as all dates except the last `num_valid_dates` dates
    train_dates = dates[:-num_valid_dates]

else:
    
    print("for any random data")
    chunk = pd.read_csv(global_input_path_train)
    
    # For example, 100 rows

    ## test data set - as last 1 lakh data points and rest above as training-
    test = pd.read_csv(global_input_path_test)

    unprocessed_test = test.copy()
    # Convert 'datetime' to datetime format
    test["datetimes"] = pd.to_datetime(test["datetime"], errors="coerce")

    # Check for any invalid values in the datetime column
    if test["datetimes"].isnull().any():
        print("Warning: Some entries in the 'datetime' column could not be parsed!")
        print(
            test[test["datetimes"].isnull()]
        )  # Display rows with invalid datetime values

    # Proceed with further processing
    test["date"] = test["datetimes"].dt.date  # Extract the date part
    test["date_id"] = (
        test["date"].astype("category").cat.codes
    )  
    # Create unique IDs for each date

    print(" all files imported")
    df = chunk.copy().iloc[: len(chunk) - 100000]
    
    df["datetimes"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Check for any invalid values in the datetime column
    if df["datetimes"].isnull().any():
        print("Warning: Some entries in the 'datetime' column could not be parsed!")
        print(
            df[test["datetimes"].isnull()]
        )  # Display rows with invalid datetime values

    # Proceed with further processing
    
    df["date"] = df["datetimes"].dt.date  # Extract the date part
    df["date_id"] = (
        df["date"].astype("category").cat.codes
    )  # Create unique IDs for each date

    dates = df["datetime"].unique()
    valid_dates = dates[-num_valid_dates:]

    # Define training dates as all dates except the last `num_valid_dates` dates
    train_dates = dates[:-num_valid_dates]
    
    
print("files are converted in to df ended.")
# In[3]: target variable defining

## defining the target for the model to predict as the return in the next 15 min

df["Target"] = df["close"].shift(-15)  # Shift the 'Close' column by -15 steps
df["Target"] = (
    (df["Target"] - df["close"]) / df["close"]
) * 100  # Compute percentage return

# Handle NaN values for rows that don't have a full future window
df["Target"] = df["Target"].fillna(0)

test["Target"] = test["close"].shift(-15)  # Shift the 'Close' column by -15 steps
test["Target"] = (
    (test["Target"] - test["close"]) / test["close"]
) * 100  # Compute percentage return

# Handle NaN values for rows that don't have a full future window
test["Target"] = test["Target"].fillna(0)
print("target variable done")


# In[4]: function for preprocessing data


def preprocess_data(df):

    # Basic preprocessing steps (example)
    """
    Preprocess the data by calculating various technical indicators.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the stock data.

    Returns
    -------
    pandas.DataFrame
        The preprocessed DataFrame with the calculated technical indicators.
    """

    stock = df
    stock["Close"] = stock["close"]
    stock["Open"] = stock["open"]
    stock["High"] = stock["high"]
    stock["Low"] = stock["low"]
    stock["Volume"] = stock["volume"]
    data = stock

    del stock

    windows = [5, 10, 15, 20]
    for i in windows:
        # Calculate the 10-day SMA
        data[f"{i}_SMA"] = data["Close"].rolling(window=i, min_periods=i).mean()

    for i in windows:

        weights = np.arange(1, i + 1)

        # Function to calculate WMA for each rolling window
        def calc_wma(values):
            return np.dot(values, weights) / weights.sum()

        # Calculate the 10-day WMA using the custom function
        data[f"{i}_WMA"] = (
            data["Close"].rolling(window=i, min_periods=i).apply(calc_wma, raw=True)
        )

    for i in windows:

        # Calculate the smoothing factor (alpha)
        alpha = 2 / (i + 1)

        # Function to calculate EMA
        def calculate_ema(series, alpha):
            ema = series.ewm(span=i, adjust=False).mean()
            return ema

        # Calculate the 10-day EMA
        data[f"{i}_EMA"] = calculate_ema(data["Close"], alpha)

    window = 5

    def calculate_stochastic_k(data, window):
        # Calculate the rolling high and low from the Close over the specified window
        high_roll = data["Close"].rolling(window=i, min_periods=1).max()
        low_roll = data["Close"].rolling(window=i, min_periods=1).min()

        # Calculate the %K using the typical formula
        stochastic_k = (data["Close"] - low_roll) / (high_roll - low_roll) * 100

        # Apply EMA to the %K values

        return stochastic_k

    data["Stochastic_K"] = calculate_stochastic_k(data, window)

    window = 5

    def calculate_stochastic_d(stochastic_k, window):
        # Apply EMA to the %K EMA values to get %D
        stochastic_d = stochastic_k.ewm(span=window, adjust=False).mean()
        return stochastic_d

    # Assuming your DataFrame is named 'data' and contains 'Close' column

    data["Stochastic_D"] = calculate_stochastic_d(data["Stochastic_K"], 5)

    rsi_window = 14  # Typically, a 14-period window is used

    # Function to calculate RSI using EMA
    def calculate_rsi(data, window):
        # Calculate the price changes
        delta = data["Close"].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate the exponential moving averages (EMA) of gains and losses
        avg_gain = gains.ewm(span=window, adjust=False).mean()
        avg_loss = losses.ewm(span=window, adjust=False).mean()

        # Calculate the Relative Strength (RS)
        rs = avg_gain / avg_loss

        # Calculate the RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    data["RSI"] = calculate_rsi(data, rsi_window)

    # macd-
    # Define the window sizes for MACD calculation
    short_window_1 = 5  # 12-period for the short-term EMA
    long_window_1 = 15  # 26-period for the long-term EMA
    # 9-period for the signal line
    short_window_2 = 5
    long_window_2 = 20

    short_window_3 = 5
    long_window_3 = 10

    # Function to calculate MACD
    def calculate_macd(data, short_window, long_window):
        # Calculate the short-term EMA
        short_ema = data["Close"].ewm(span=short_window, adjust=False).mean()

        # Calculate the long-term EMA
        long_ema = data["Close"].ewm(span=long_window, adjust=False).mean()

        # Calculate the MACD line (difference between short-term EMA and long-term EMA)
        macd = short_ema - long_ema
        return macd

    # Assuming your DataFrame is named 'data' and contains 'Close' column
    data["MACD_1"] = calculate_macd(data, short_window_1, long_window_1)
    data["MACD_2"] = calculate_macd(data, short_window_2, long_window_2)
    data["MACD_3"] = calculate_macd(data, short_window_3, long_window_3)

    # larry williams--
    for i in windows:
        # Define the window size for Williams %R calculation
        williams_r_window = i  # Typically, a 14-period window is used

        # Function to calculate Williams %R using Close
        def calculate_williams_r(data, window):
            # Calculate the highest price over the window
            high_n = data["Close"].rolling(window=window).max()

            # Calculate the lowest price over the window
            low_n = data["Close"].rolling(window=window).min()

            # Calculate Williams %R
            williams_r = ((high_n - data["Close"]) / (high_n - low_n)) * -100

            return williams_r

        # Assuming your DataFrame is named 'data' and contains 'Close' column
        data[f"Williams_%R_{i}"] = calculate_williams_r(data, williams_r_window)

    def calculate_rdp(data, x):
        # Calculate the moving average (CLt-x)
        moving_average = data["Close"].rolling(window=x).mean()

        # Calculate RDP (RDPt)
        rdp = ((data["Close"] - moving_average) / moving_average) * 100

        return rdp

    for i in windows:

        data[f"RDP_{i}"] = calculate_rdp(data, i)

    # Bias
    def calculate_bias(data, interval):
        # Calculate the moving average
        moving_average = data["Close"].rolling(window=interval).mean()

        # Calculate Bias
        bias = (data["Close"] - moving_average) / moving_average

        return bias

    # You can change this to any desired interval
    for i in windows:
        # Calculate Bias and add it as a new column in the DataFrame
        data[f"Bias_{i}"] = calculate_bias(data, i)

    def calculate_mtm(data, interval):

        moving_average = data["Close"].rolling(window=interval).mean()

        # Calculate Momentum (MTM)
        mtm = data["Close"] - moving_average

        return mtm

    for i in windows:
        # You can change this to any desired interval

        # Calculate MTM and add it as a new column in the DataFrame
        data[f"MTM_{i}"] = calculate_mtm(data, i)

    def calculate_roc(data, interval):
        # Calculate the moving average of 'Close'
        moving_average = data["Close"].rolling(window=interval).mean()

        # Calculate Rate of Change (ROC)
        roc = data["Close"] / moving_average

        return roc

    for i in windows:
        # Calculate ROC and add it as a new column in the DataFrame
        data[f"ROC_{i}"] = calculate_roc(data, i)

    def calculate_oscp(data, x_interval, y_interval):
        # Calculate the moving average for both columns
        ma_x = data["Close"].rolling(window=x_interval).mean()
        ma_y = data["Close"].rolling(window=y_interval).mean()

        # Calculate OSCP
        oscp = (ma_x - ma_y) / ma_x

        return oscp

    # Calculate OSCP and add it as a new column in the DataFrame
    data["OSCP_1"] = calculate_oscp(data, short_window_1, long_window_1)
    data["OSCP_2"] = calculate_oscp(data, short_window_2, long_window_2)
    data["OSCP_3"] = calculate_oscp(data, short_window_3, long_window_3)

    def calculate_median_price(data, period):
        # Calculate the highest high and lowest low over the specified period
        highest_high = data["High"].rolling(window=period).max()
        lowest_low = data["Low"].rolling(window=period).min()

        # Calculate Median Price
        median_price = (highest_high + lowest_low) / 2

        return median_price

    # Define the period for calculation
    period = 10  # You can change this to any desired period

    # Calculate Median Price and add it as a new column in the DataFrame
    data[f"Median_Price_{period}"] = calculate_median_price(data, period)

    # #print the DataFrame info to check the added column
    # print(data.info())

    data["highest_high"] = data["High"].rolling(window=period).max()
    data["lowest_low"] = data["Low"].rolling(window=period).min()

    # def calculate_cci(data, period):
    # # Use Closing Price instead of Typical Price
    #     data['Closing_Price'] = data['Close']

    #     # Calculate SMA of Closing Price
    #     data['SMA'] = data['Closing_Price'].rolling(window=period).mean()

    #     # Calculate Mean Deviation
    #     data['Mean_Deviation'] = data['Closing_Price'].rolling(window=period).apply(lambda x: (abs(x - x.mean())).mean())

    #     # Calculate CCI
    #     data[f'CCI{period}'] = (data['Closing_Price'] - data['SMA']) / (0.015 * data['Mean_Deviation'])

    #     return data

    # Define the period for CCI calculation
    # for i in windows:
    #     data = calculate_cci(data, i)

    def calculate_signal_line(data, period_x, period_y):
        # Calculate SMA for x and y
        data[f"SMA_{period_x}"] = data["Close"].rolling(window=period_x).mean()
        data[f"SMA_{period_y}"] = data["Close"].rolling(window=period_y).mean()

        # Calculate the Signal Line
        data["Signal_Line"] = (1 / (10 * data[f"SMA_{period_y}"])) * (
            data[f"SMA_{period_x}"] - data[f"SMA_{period_y}"]
        ) + data[f"SMA_{period_y}"]
        return data

    # Define the periods for SMA calculation
    period_x = 10  # Period for SMA of x (you can change this)
    period_y = 20  # Period for SMA of y (you can change this)

    # Calculate the Signal Line and add it to the DataFrame
    data = calculate_signal_line(data, period_x, period_y)

    def calculate_ultimate_oscillator(data, short_period, mid_period, long_period):
        # Calculate moving averages over different periods
        avg_x = data["Close"].rolling(window=short_period).mean()
        avg_y = data["Close"].rolling(window=mid_period).mean()
        avg_z = data["Close"].rolling(window=long_period).mean()

        # Calculate the Ultimate Oscillator
        data["UO"] = 100 * (1 / (4 + 2 + 1)) * (4 * avg_x + 2 * avg_y + avg_z)

        return data

    # Define the periods for short, mid, and long-term moving averages
    short_period = 10  # Short-term moving average period
    mid_period = 20  # Mid-term moving average period
    long_period = 30  # Long-term moving average period

    # Calculate the Ultimate Oscillator and add it to the DataFrame
    data = calculate_ultimate_oscillator(data, short_period, mid_period, long_period)

    def calculate_bp_tr(data):

        data["BP"] = data["Close"] - data[["Low", "Close"]].shift(1).min(axis=1)

        # True Range (TR)
        data["TR"] = data[["High", "Close"]].shift(1).max(axis=1) - data[
            ["Low", "Close"]
        ].shift(1).min(axis=1)

        return data

    # Function to calculate Ultimate Oscillator (UO)
    def calculate_ultimate_oscillator(
        data, short_period=7, medium_period=14, long_period=28
    ):
        # Calculate BP and TR
        data = calculate_bp_tr(data)

        # Rolling averages of BP/TR for different timeframes
        avg_x = (
            data["BP"].rolling(window=short_period).sum()
            / data["TR"].rolling(window=short_period).sum()
        )
        avg_y = (
            data["BP"].rolling(window=medium_period).sum()
            / data["TR"].rolling(window=medium_period).sum()
        )
        avg_z = (
            data["BP"].rolling(window=long_period).sum()
            / data["TR"].rolling(window=long_period).sum()
        )

        # Ultimate Oscillator formula
        data["UO"] = 100 * ((4 * avg_x) + (2 * avg_y) + (1 * avg_z)) / (4 + 2 + 1)

        return data

    # Define periods for short, medium, and long timeframes
    short_period = 10
    medium_period = 20
    long_period = 30

    # Calculate Ultimate Oscillator and add it to the DataFrame
    data = calculate_ultimate_oscillator(data, short_period, medium_period, long_period)

    def calculate_ulcer_index(data, window):
        # Calculate the highest high over the window (HH)
        hh = data["Close"].rolling(window=window).max()

        # Calculate Rt(x)
        rt = 100 * (data["Close"] - hh) / hh

        # Square the Rt values (as per the formula)
        rt_sq = rt**2

        # Ulcer Index (root of the average squared drawdowns)
        data["Ulcer_Index"] = np.sqrt(rt_sq.rolling(window=window).mean())

        return data

    # Define the window for the Ulcer Index calculation
    ulcer_window = 10  # You can change this window

    # Calculate the Ulcer Index and add it to the DataFrame
    data = calculate_ulcer_index(data, ulcer_window)

    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    # Function to calculate TSI
    def calculate_tsi(data, window):
        # Calculate the Momentum (MTM)
        mtm = data["Close"] - data["Close"].shift(1)

        # Apply EMA to the MTM for the first smoothing
        ema_mtm = ema(mtm, window)

        # Apply a second EMA to the already smoothed MTM
        Double_EMA_MTM = ema(ema_mtm, window)

        # Calculate the absolute value of MTM and apply EMA twice
        Abs_MTM = mtm.abs()
        EMA_Abs_MTM = ema(Abs_MTM, window)
        Double_EMA_Abs_MTM = ema(EMA_Abs_MTM, window)

        # Calculate the TSI
        data["TSI"] = 100 * (Double_EMA_MTM / Double_EMA_Abs_MTM)

        return data

    # Define the window for the TSI calculation
    tsi_window = 25  # Typical value is 25, but you can adjust

    # Calculate TSI and add it to the DataFrame
    data = calculate_tsi(data, tsi_window)

    def calculate_ad_oscillator(data):

        # Calculate the A/D oscillator value using the formula (HighPrice - ClosePrice(t-1)) / (HighPrice - LowPrice)
        data["A/D Oscillator"] = (data["High"] - data["Close"].shift(1)) / (
            data["High"] - data["Low"]
        )
        return data

    # Calculate the A/D Oscillator
    data = calculate_ad_oscillator(data)

    from sklearn.preprocessing import StandardScaler

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the data
    cols = list(data.columns)  # Convert to a list
    cols = [
        col for col in cols if col not in ["datetime", "datetimes", "date", "date_id"]
    ]  # Remove unwanted columns
    data[cols] = data[cols].replace([np.inf, -np.inf], np.nan)
    data[cols] = data[cols].fillna(0)
    standardized_data = scaler.fit_transform(data[cols])

    # Create a new DataFrame with standardized values

    data_df = pd.DataFrame(standardized_data, columns=cols)
    data_df["datetime"] = data["datetime"]
    data_df["date_id"] = data["date_id"]

    return data_df


# In[5]: -- data pre processing

print("data pre-processing started")
df = preprocess_data(df)
print(df.head())


test = preprocess_data(test)
print(test.head())

print(df.shape)

# test.drop(["datetimes", "date"], axis=1, inplace=True)

print("data pre-processing done")


# In[6]: -- model defining


from sklearn.model_selection import train_test_split

os.system("mkdir models")
print("started")
# Define the path to load pre-trained models (if not in training mode)
model_path = "models"

# If in training mode, prepare validation data
if TRAINING:
    print("training data prep")

    # Extract features, target, and weights for validation dates
    y_valid = df["Target"]
    # df.drop(["Target"], axis =1, inplace = True)
    cols = df.columns

    X_valid = df[cols]
    X_valid.drop(["Target", "datetime", "date_id"], axis=1, inplace=True)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_valid, y_valid, test_size=0.2, random_state=42
    )

    # Sample 500,000 rows from X_train
    samples = 500000
    X_train = X_train.sample(samples, replace=False, random_state=None)

    # Now get the corresponding rows from y_train
    y_train = y_train.loc[X_train.index]


# Initialize a list to store trained models
modelss = []


# Function to train a model or load a pre-trained model
def train(model_dict, model_name="lgb"):
    """
    Train a machine learning model using the specified configuration.

    Parameters
    ----------
    model_dict : dict
        A dictionary containing model configurations for LightGBM, XGBoost, and CatBoost.
    model_name : str, optional
        The name of the model to train ('lgb', 'xgb', or 'cbt'), by default 'lgb'.

    Description
    -----------
    This function handles the training and validation of machine learning models. It selects
    training data based on fold number, preprocesses features and targets, and trains the model
    with early stopping and evaluation logging. The trained model is saved to disk, and models
    are loaded from disk if not in training mode.
    """
    print(f"Training {model_name} model for fold {i}...")

    if TRAINING:
        # Select dates for training based on the fold number
        print("Selecting dates for training...")
        selected_dates = [
            date for ii, date in enumerate(train_dates) if ii % N_fold != i
        ]

        # Get the model from the dictionary
        print("Getting the model from the dictionary...")
        model = model_dict[model_name]

        # Extract features, target, and weights for the selected training dates
        print(
            "Extracting features, target, and weights for the selected training dates..."
        )
        y_valid = df["Target"]
        cols = df.columns
        X_valid = df[cols]
        X_valid.drop(["Target", "datetime", "date_id"], axis=1, inplace=True)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_valid, y_valid, test_size=0.2, random_state=42
        )

        print("Sampling 500,000 rows from X_train...")
        samples = len(df)//2
        X_train = X_train.sample(samples, replace=False, random_state=None)

        # Now get the corresponding rows from y_train
        print("Getting the corresponding rows from y_train...")
        y_train = y_train.loc[X_train.index]

        # Train the model
        if model_name == "lgb":
            # Train LightGBM model with early stopping and evaluation logging
            model.fit(
                X_train,
                y_train,
                eval_metric=[r2_lgb],
                eval_set=[(X_valid, y_valid)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(10)],
            )

        elif model_name == "cbt":
            # Prepare evaluation set for CatBoost
            evalset = cbt.Pool(X_valid, y_valid)

            # Train CatBoost model with early stopping and verbose logging
            model.fit(
                evalset,
                use_best_model=True,  # Keeps the best model based on validation set
                verbose=200,  # Logs progress every 100 iterations
            )

        else:
            # Train XGBoost model with early stopping and verbose logging
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=10,
                early_stopping_rounds=200,
            )

        # Append the trained model to the list
        modelss.append(model)

        # Save the trained model to a file
        joblib.dump(model, f"./models/{model_name}_{i}.model")

        # Collect garbage to free up memory
        import gc

        gc.collect()

    else:
        # If not in training mode, load the pre-trained model from the specified path
        modelss.append(joblib.load(f"{model_path}/{model_name}_{i}_individual.model"))

    return


# Custom R2 metric for XGBoost
def r2_xgb(y_true, y_pred):
    r2 = 1 - np.mean((y_pred - y_true) ** 2) / (np.mean(y_true**2) + 1e-38)
    return "r2", -r2


# Custom R2 metric for LightGBM
def r2_lgb(y_true, y_pred):
    r2 = 1 - np.mean((y_pred - y_true) ** 2) / (np.mean(y_true**2) + 1e-38)
    return "r2", r2, True  # Return True to maximize the metric


# Custom R2 metric for CatBoost

# Dictionary to store different models with their configurations
model_dict = {
    "lgb": lgb.LGBMRegressor(
        n_estimators=500 * 2, device="gpu", gpu_use_dp=True, objective="l2"
    ),
    "xgb": xgb.XGBRegressor(
        n_estimators=2000 ,
        learning_rate=0.1,
        max_depth=6,
        tree_method="hist",
        device="cuda",
        objective="reg:squarederror",
        eval_metric="rmse",
        disable_default_eval_metric=True,
    ),
    "cbt": cbt.CatBoostRegressor(
        iterations=1000 ,
        learning_rate=0.05,
        task_type="GPU",
        loss_function="RMSE",
        eval_metric="R2",
    ),
}

# In[7]: training the models

print("calling the train fns")
# Train models for each fold
if(stock=="BTC"):
    for i in range(N_fold):
        print(f"{i} th fold training ")
        # print("lgb")
        # train(model_dict, "lgb")
        print("xgb")
        train(model_dict, "xgb")
        print("cbt")
        train(model_dict, "cbt")

elif(stock=="ETH"):
     for i in range(N_fold):
        print(f"{i} th fold training ")
        print("lgb")
        train(model_dict, "lgb")
        print("xgb")
        train(model_dict, "xgb")
        print("cbt")
        train(model_dict, "cbt")
else:
    for i in range(N_fold):
        print(f"{i} th fold training ")
        print("lgb")
        train(model_dict, "lgb")
        print("xgb")
        train(model_dict, "xgb")
        print("cbt")
        train(model_dict, "cbt")
        

print("model training ended")


# In[8]: loading the models and predicting on the test data-


y_test = test["Target"][:test_data_points]  # Take the first 10,000 points only
print("Started")

# Prepare the feature set
X_test = test.iloc[:test_data_points].drop(["Target", "datetime", "date_id"], axis=1)

pred = np.array(
    [model.predict(X_test) for model in modelss]
)  # Shape: (n_models, 10000)
# Average predictions across all models
pred_mean = np.mean(pred, axis=0)  # Shape: (10000,)

# Calculate R² score for the first 10,000 points
r2 = r2_score(y_test, pred_mean)
print(f"R² Score: {r2}")
# print(models)

print(pred)


# In[9]:saving the predictions

colss = ["open", "high", "close", "low", "volume"]
X_test = X_test[colss]
unprocessed_test = unprocessed_test.iloc[:test_data_points]
unprocessed_test["target_returns"] = y_test
unprocessed_test["predicted_mean"] = pred_mean
X_test["Target_returns"] = y_test
X_test["predicted_mean"] = pred_mean
print(X_test.head())

# X_test.to_csv("X_test_predicted.csv", index=False)
unprocessed_test.to_csv(f"{stock}_test_predicted.csv", index=False)
