import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Path helpers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Gradient boosting Stocks")

# Indices in the 10-column scaler (5 close, 5 volume)
TICKER_MAP = {
    'AAPL': 0,
    'AMZN': 1,
    'GOOGL': 2,
    'MSFT': 3,
    'TSLA': 4
}

def train_stock_model(ticker):
    print(f"\n--- Training XGBoost Model for {ticker} ---")
    feat_path = os.path.join(DATA_DIR, "stocks_features_advanced.csv")
    if not os.path.exists(feat_path):
        print(f"Error: {feat_path} logic not found.")
        return

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    
    # Focus on recent data (2022+) for higher accuracy in current volatile markets
    df = df[df.index >= '2022-01-01']
    
    target_col = f'{ticker}_Target'
    if target_col not in df.columns:
        print(f"Error: {target_col} not found in features.")
        return

    y = df[target_col]
    # Features X are everything EXCEPT the target columns
    X = df.drop([col for col in df.columns if 'Target' in col], axis=1)
    
    # Time-series split (no shuffle)
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # XGBoost Parameters tuned for stocks
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Save Model
    model_filename = f"{ticker.lower()}_xgboost_model.pkl"
    joblib.dump(model, os.path.join(BASE_DIR, model_filename))
    print(f"Saved {model_filename}")

    # Evaluation on the last day
    preds = model.predict(X_test)
    
    # Load Scaler
    scaler_path = os.path.join(DATA_DIR, "stocks_scaler.pkl")
    scaler = joblib.load(scaler_path)
    
    def to_real_dollars(val_scaled, t_name):
        # The scaler was fitted on 10 columns (5 Close, 5 Volume)
        dummy = np.zeros((1, 10))
        dummy[0, TICKER_MAP[t_name]] = val_scaled
        log_price = scaler.inverse_transform(dummy)[0, TICKER_MAP[t_name]]
        return np.expm1(log_price)

    last_pred_dollars = to_real_dollars(preds[-1], ticker)
    last_actual_dollars = to_real_dollars(y_test.iloc[-1], ticker)
    
    accuracy = (1 - abs(last_pred_dollars - last_actual_dollars) / last_actual_dollars) * 100
    
    print(f"Prediction for tomorrow's {ticker}: ${last_pred_dollars:,.2f}")
    print(f"Actual price: ${last_actual_dollars:,.2f}")
    print(f"Recent Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    for ticker in TICKER_MAP.keys():
        train_stock_model(ticker)
