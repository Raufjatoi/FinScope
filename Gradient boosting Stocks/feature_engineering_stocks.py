import pandas as pd
import numpy as np
import os

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_advanced_features():
    print("Generating advanced technical indicators for Stocks...")
    raw_scaled_path = 'stocks_data_scaled.csv'
    if not os.path.exists(raw_scaled_path):
        print("Error: scaled data not found.")
        return

    df = pd.read_csv(raw_scaled_path, index_col=0, parse_dates=True)
    
    # Identify unique tickers (Close columns)
    tickers = [col.replace('_Close', '') for col in df.columns if '_Close' in col]
    final_df = df.copy()
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        close = df[f'{ticker}_Close']
        volume = df[f'{ticker}_Volume']
        
        # 1. RSI
        final_df[f'{ticker}_RSI'] = calculate_rsi(close)
        
        # 2. Bollinger Bands
        ma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        final_df[f'{ticker}_BB_Upper'] = ma20 + (std20 * 2)
        final_df[f'{ticker}_BB_Lower'] = ma20 - (std20 * 2)
        
        # 3. Lags
        for i in range(1, 4):
            final_df[f'{ticker}_Close_Lag_{i}'] = close.shift(i)
            final_df[f'{ticker}_Volume_Lag_{i}'] = volume.shift(i)
            
        # 4. Target Label: NEXT DAY Price
        final_df[f'{ticker}_Target'] = close.shift(-1)

    # Drop the rows with NaN
    final_df = final_df.dropna()
    
    final_df.to_csv('stocks_features_advanced.csv')
    print(f"Complete. Generated {final_df.shape[1]} columns. Saved to stocks_features_advanced.csv")

if __name__ == "__main__":
    create_advanced_features()
