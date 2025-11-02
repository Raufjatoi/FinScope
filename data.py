import yfinance as yf
import pandas as pd

# --- Stocks & Crypto symbols ---
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "DOGE-USD"]

# --- Download stock data ---
stock_data = yf.download(stocks, start="2023-01-01", end="2025-01-01", group_by='ticker')

# --- Download crypto data ---
crypto_data = yf.download(cryptos, start="2023-01-01", end="2025-01-01", group_by='ticker')

# --- Save to CSV files ---
stock_data.to_csv("stocks_data.csv")
crypto_data.to_csv("crypto_data.csv")

print("✅ Stocks and crypto data saved successfully!")