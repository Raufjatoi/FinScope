import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_advanced_features():
    print("Generating advanced technical indicators...")
    df = pd.read_csv('crypto_data_scaled.csv', index_col=0, parse_dates=True)
    
    # Identify unique tickers (e.g., BTC-USD from BTC-USD_Close)
    tickers = [col.replace('_Close', '') for col in df.columns if '_Close' in col]
    final_df = df.copy()
    
    for ticker in tickers:
        close = df[f'{ticker}_Close']
        volume = df[f'{ticker}_Volume']
        
        # 1. RSI (Momentum Indicator)
        final_df[f'{ticker}_RSI'] = calculate_rsi(close)
        
        # 2. Bollinger Bands (Volatility/Range Indicator)
        ma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        final_df[f'{ticker}_BB_Upper'] = ma20 + (std20 * 2)
        final_df[f'{ticker}_BB_Lower'] = ma20 - (std20 * 2)
        
        # 3. Lags (Yesterday, 2 days ago, 3 days ago)
        for i in range(1, 4):
            final_df[f'{ticker}_Close_Lag_{i}'] = close.shift(i)
            final_df[f'{ticker}_Volume_Lag_{i}'] = volume.shift(i)
            
        # 4. Target Label: TOMORROW'S Price (What we want to predict)
        final_df[f'{ticker}_Target'] = close.shift(-1)

    # Drop the rows with NaN created by indicators and targets
    final_df = final_df.dropna()
    
    final_df.to_csv('crypto_features_advanced.csv')
    print(f"Complete. Generated {final_df.shape[1]} features and targets.")

if __name__ == "__main__":
    create_advanced_features()
