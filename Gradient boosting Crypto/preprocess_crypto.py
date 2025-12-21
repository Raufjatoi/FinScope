import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data():
    print("Starting Log-Transformed preprocessing...")
    df = pd.read_csv('crypto_data_raw.csv', header=[0, 1], index_col=0)
    
    close_df = df['Close'].copy()
    volume_df = df['Volume'].copy()
    
    close_df.columns = [f"{col}_Close" for col in close_df.columns]
    volume_df.columns = [f"{col}_Volume" for col in volume_df.columns]
    
    # 1. Apply Log Transformation to Prices
    # We use log1p (log(1+x)) to be safe with any small values
    # Prices in log space are much easier for models to predict over 10 years
    close_log = np.log1p(close_df.fillna(0))
    volume_log = np.log1p(volume_df.fillna(0))
    
    combined_df = pd.concat([close_log, volume_log], axis=1)
    combined_df = combined_df.dropna(how='all')
    
    combined_df.to_csv('crypto_data_cleaned.csv')
    
    # 2. Scale the Log-Values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_df)
    
    scaled_df = pd.DataFrame(scaled_data, index=combined_df.index, columns=combined_df.columns)
    scaled_df.to_csv('crypto_data_scaled.csv')
    
    joblib.dump(scaler, 'crypto_scaler.pkl')
    print("Log transformation and scaling complete.")

if __name__ == "__main__":
    preprocess_data()
