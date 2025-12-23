import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def preprocess_stocks():
    print("Starting Log-Transformed preprocessing for Stocks...")
    raw_path = 'stocks_data_raw.csv'
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    # Load raw data
    print("Loading raw CSV...")
    df = pd.read_csv(raw_path, header=[0, 1], index_col=0)
    print(f"Loaded data with shape: {df.shape}")
    
    # Extract only Close and Volume
    try:
        close_df = df['Close'].copy()
        volume_df = df['Volume'].copy()
    except KeyError as e:
        print(f"Error extracting Close/Volume: {e}")
        print("Available columns:", df.columns.levels[0])
        return

    # Fill NaNs using forward fill then backward fill (standard for time series)
    print("Filling missing values...")
    close_df = close_df.ffill().bfill()
    volume_df = volume_df.ffill().bfill()
    
    # Prefix columns
    close_df.columns = [f"{col}_Close" for col in close_df.columns]
    volume_df.columns = [f"{col}_Volume" for col in volume_df.columns]
    
    # Combine
    combined_df = pd.concat([close_df, volume_df], axis=1)
    print(f"Combined shape: {combined_df.shape}")

    # 1. Apply Log Transformation
    print("Applying Log Transformation...")
    combined_log = np.log1p(combined_df)
    
    # Save cleaned logs
    combined_log.to_csv('stocks_data_cleaned.csv')
    print("Cleaned data saved to stocks_data_cleaned.csv")
    
    # 2. Scale the Log-Values
    print("Scaling values (MinMaxScaler)...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_log)
    
    scaled_df = pd.DataFrame(scaled_data, index=combined_log.index, columns=combined_log.columns)
    scaled_df.to_csv('stocks_data_scaled.csv')
    print("Scaled data saved to stocks_data_scaled.csv")
    
    # Save the scaler
    joblib.dump(scaler, 'stocks_scaler.pkl')
    print("Scaler saved to stocks_scaler.pkl")

if __name__ == "__main__":
    preprocess_stocks()
