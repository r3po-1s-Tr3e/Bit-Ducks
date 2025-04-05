import requests
import pandas as pd
import time

# Binance Futures API endpoint
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"

# Parameters
symbol = "BTCUSDT"
intervals = ["1m","4h","1d"]
limit = 1000  # Max limit per request
start_time = int(pd.Timestamp("2020-01-01").timestamp() * 1000)
end_time = int(pd.Timestamp("2024-12-31").timestamp() * 1000)  # Until Dec 2023

# Empty list to store data
all_klines = []

for interval in intervals:
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
          response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
          data = response.json()

          if isinstance(data, list) and data:
              # Check if the first Kline has the expected number of elements
              if len(data[0]) == 12:
                  # Append data
                  all_klines.extend(data)

                  # Update start_time for the next batch
                  start_time = data[-1][0] + 1  # Move to next candle

                  print(f"Fetched {len(data)} candles, Next Start: {pd.to_datetime(start_time, unit='ms')}")
              else:
                  print(f"Unexpected data format: {data}")
                  break  # Stop if the data format is not as expected
          elif isinstance(data, dict) and "code" in data:
              print(f"Binance API Error: {data}")
              break  # Stop if it's a Binance API error
          elif not data:
              print("No more data received.")
              break  # Stop if no more data
          else:
              print(f"Unexpected response: {data}")
              break

          time.sleep(0.5)  # Sleep to avoid rate limits

      except requests.exceptions.RequestException as e:
          print(f"Request Error: {e}")
          break
      except Exception as e:
          print("Error:", e)
          break

  # Convert to DataFrame
  if all_klines:
      df = pd.DataFrame(all_klines, columns=[
          "Open Time", "Open", "High", "Low", "Close", "Volume",
          "Close Time", "Quote Asset Volume", "Trades",
          "Taker Base Volume", "Taker Quote Volume", "Ignore"
      ])

      # Convert timestamps
      df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
      df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")

      # Save to CSV
      df.to_csv(f'BTCUSDT_{interval}_all_data.csv', index=False)

      print(f"Data saved to BTCUSDT_{interval}_all_data.csv")
  else:
      print("No Kline data was collected.")