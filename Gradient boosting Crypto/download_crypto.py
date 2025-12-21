import yfinance as yf
import pandas as pd
from datetime import datetime

def download_crypto_data():
    # Tickers for major cryptocurrencies
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'DOGE-USD']
    
    start_date = "2015-01-01"
    # Set end date to current or specific 2025 date if in future
    end_date = "2025-01-01" 
    
    print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date, interval='1d')
    
    # Check if data is empty
    if data.empty:
        print("Error: No data downloaded. Check your internet connection or tickers.")
        return

    # Save raw data
    filename = 'crypto_data_raw.csv'
    data.to_csv(filename)
    print(f"Successfully saved raw data to {filename}")
    print(data.head())

if __name__ == "__main__":
    download_crypto_data()
