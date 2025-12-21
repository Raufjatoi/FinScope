import pandas as pd
import numpy as np
import joblib
import os
try:
    import xgboost as xgb
except ImportError:
    print("Installing xgboost...")
    os.system('pip install xgboost')
    import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Path helpers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Gradient boosting Crypto")

def train_xgboost():
    print("Loading data from Gradient Boosting folder...")
    # We reuse the advanced features we already generated
    feat_path = os.path.join(DATA_DIR, "crypto_features_advanced.csv")
    if not os.path.exists(feat_path):
        print(f"Error: Could not find features at {feat_path}")
        return

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    
    # FOCUS ON RECENT DATA (Last 3 Years: 2022-2025)
    # This ignores the 'noise' from 2015-2021 where BTC was much cheaper
    df = df[df.index >= '2022-01-01']
    print(f"Focused on recent data: {len(df)} days of price action.")

    target_col = 'BTC-USD_Target'
    y = df[target_col]
    X = df.drop([col for col in df.columns if 'Target' in col], axis=1)
    
    # Chronological Split
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training XGBoost on {len(X_train)} samples...")
    
    # XGBoost Parameters for Finance
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01, # Lower learning rate for more stable learning
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              verbose=False)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"XGBoost Model RMSE (Log Space): {rmse:.6f}, R2 Score: {r2:.4f}")
    
    joblib.dump(model, os.path.join(BASE_DIR, 'btc_xgboost_model.pkl'))
    
    # --- UN-NORMALIZATION ---
    scaler_path = os.path.join(DATA_DIR, "crypto_scaler.pkl")
    scaler = joblib.load(scaler_path)
    
    latest_today = X_test.iloc[-1:]
    pred_scaled = model.predict(latest_today)[0]
    actual_scaled = y_test.iloc[-1]
    
    def to_real_price(val_scaled):
        dummy = np.zeros((1, 10))
        dummy[0, 1] = val_scaled
        log_price = scaler.inverse_transform(dummy)[0, 1]
        return np.expm1(log_price)

    predicted_price = to_real_price(pred_scaled)
    actual_price = to_real_price(actual_scaled)

    print("\n--- XGBoost BTC Prediction (2022-2025 Focus) ---")
    print(f"Predicted Price for Tomorrow: ${predicted_price:,.2f}")
    print(f"Actual Price Tomorrow:        ${actual_price:,.2f}")
    
    error = abs(predicted_price - actual_price)
    accuracy_pct = (1 - error / actual_price) * 100
    print(f"Accuracy:                     {accuracy_pct:.2f}%")

if __name__ == "__main__":
    train_xgboost()
