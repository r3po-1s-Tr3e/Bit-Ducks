import requests
import pandas as pd
import time

# Binance Futures API endpoint
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"

# Parameters
symbol = "BTCUSDT"
interval = "4h"
limit = 1000  # Max limit per request
start_time = int(pd.Timestamp("2020-01-01").timestamp() * 1000)
end_time = int(pd.Timestamp("2023-12-31").timestamp() * 1000)  # Until Dec 2024

# Empty list to store data
all_klines = []

while start_time < end_time:
    try:
        # Request Kline data
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if not data:
            break  # Stop if no more data

        # Append data
        all_klines.extend(data)

        # Update start_time for the next batch
        start_time = data[-1][0] + 1  # Move to next candle

        print(f"Fetched {len(data)} candles, Next Start: {pd.to_datetime(start_time, unit='ms')}")
        time.sleep(0.5)  # Sleep to avoid rate limits

    except Exception as e:
        print("Error:", e)
        break

# Convert to DataFrame
df = pd.DataFrame(all_klines, columns=[
    "Open Time", "Open", "High", "Low", "Close", "Volume", 
    "Close Time", "Quote Asset Volume", "Trades", 
    "Taker Base Volume", "Taker Quote Volume", "Ignore"
])

# Convert timestamps
df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")

# Save to CSV
df.to_csv(f'BTCUSDT_{interval}_ data.csv', index=False)

print(f"Data saved to BTCUSDT_{interval}_ data.csv")