import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Path helpers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Gradient boosting Crypto")

# Mapping of tickers to their column index in the scaler
# Based on how they were concatenated in preprocess_crypto.py:
# BNB, BTC, DOGE, ETH, SOL
TICKER_MAP = {
    'BNB-USD': 0,
    'BTC-USD': 1,
    'DOGE-USD': 2,
    'ETH-USD': 3,
    'SOL-USD': 4
}

def train_crypto_model(ticker):
    print(f"\n--- Training Model for {ticker} ---")
    feat_path = os.path.join(DATA_DIR, "crypto_features_advanced.csv")
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    
    # Focus on recent data (2022+)
    df = df[df.index >= '2022-01-01']
    
    # Target is exactly target for this ticker
    target_col = f'{ticker}_Target'
    if target_col not in df.columns:
        print(f"Error: {target_col} not found in features.")
        return

    y = df[target_col]
    # Features X are everything EXCEPT the targets
    X = df.drop([col for col in df.columns if 'Target' in col], axis=1)
    
    # Split
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # XGBoost Parameters
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Save Model
    model_name = f"{ticker.lower().split('-')[0]}_xgboost_model.pkl"
    joblib.dump(model, os.path.join(BASE_DIR, model_name))
    print(f"Saved {model_name}")

    # Evaluation
    preds = model.predict(X_test)
    
    # Un-normalize logic
    scaler_path = os.path.join(DATA_DIR, "crypto_scaler.pkl")
    scaler = joblib.load(scaler_path)
    
    def to_real_price(val_scaled, t_name):
        dummy = np.zeros((1, 10)) # 5 close, 5 volume
        dummy[0, TICKER_MAP[t_name]] = val_scaled
        log_price = scaler.inverse_transform(dummy)[0, TICKER_MAP[t_name]]
        return np.expm1(log_price)

    last_pred = to_real_price(preds[-1], ticker)
    last_actual = to_real_price(y_test.iloc[-1], ticker)
    
    accuracy = (1 - abs(last_pred - last_actual) / last_actual) * 100
    print(f"Prediction for tomorrow's {ticker}: ${last_pred:,.4f}")
    print(f"Actual price: ${last_actual:,.4f}")
    print(f"Recent Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    for ticker in TICKER_MAP.keys():
        train_crypto_model(ticker)
