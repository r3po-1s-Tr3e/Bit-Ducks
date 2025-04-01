import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

btc_data = pd.read_csv('BTCUSDT_1d_all_data.csv')
btc_data['Open Time'] = pd.to_datetime(btc_data['Open Time'])
btc_data.set_index('Open Time', inplace=True)
btc_data.sort_index(inplace=True)


btc_data['Returns'] = btc_data['Close'].pct_change().dropna()
btc_data.dropna(subset=['Returns'], inplace=True)
scaled_returns = btc_data['Returns'] * 100

model = arch_model(scaled_returns, p=1, q=1, vol='Garch', dist='Normal')
results = model.fit(disp='off')

print("GARCH(1,1) Model Summary:")
print(results.summary())
conditional_volatility = results.conditional_volatility

# Plot the conditional volatility
plt.figure(figsize=(12, 6))
plt.plot(conditional_volatility, label='Conditional Volatility (GARCH)')
plt.title('BTC/USD Daily Conditional Volatility (GARCH)')
plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.legend()
plt.grid(True)
plt.show()


btc_data['RV_Daily'] = btc_data['Returns']**2
btc_data['RV_Weekly'] = btc_data['RV_Daily'].rolling(window=5).mean().shift(1)
btc_data['RV_Monthly'] = btc_data['RV_Daily'].rolling(window=22).mean().shift(1)

btc_data_har = btc_data.dropna()

import statsmodels.api as sm

y_har = np.sqrt(btc_data_har['RV_Daily']) * 100 # Predict daily volatility
X_har = btc_data_har[['RV_Daily', 'RV_Weekly', 'RV_Monthly']]
X_har['Constant'] = 1

model_har = sm.OLS(y_har, X_har)
results_har = model_har.fit()

print("\nHAR Model Summary:")
print(results_har.summary())

plt.figure(figsize=(12, 6))
plt.plot(y_har.index, y_har, label='Daily Volatility (Actual)')
plt.plot(y_har.index, results_har.fittedvalues, label='Daily Volatility (HAR Predicted)')
plt.title('BTC/USD Daily Volatility vs. HAR Model Prediction')
plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.legend()
plt.grid(True)
plt.show()
