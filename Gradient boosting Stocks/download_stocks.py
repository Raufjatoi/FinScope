import yfinance as yf
import pandas as pd
import os

def download_stock_data():
    # Tickers for top 5 stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    start_date = "2015-01-01"
    end_date = "2025-01-01" 
    
    print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date, interval='1d')
    
    if data.empty:
        print("Error: No data downloaded.")
        return

    # Save raw data
    filename = 'stocks_data_raw.csv'
    data.to_csv(filename)
    print(f"Successfully saved raw data to {filename}")
    print(data.head())

if __name__ == "__main__":
    download_stock_data()
