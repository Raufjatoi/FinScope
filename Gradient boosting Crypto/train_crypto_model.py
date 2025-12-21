import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_log_model():
    print("Loading log-transformed features...")
    df = pd.read_csv('crypto_features_advanced.csv', index_col=0, parse_dates=True)
    
    target_col = 'BTC-USD_Target'
    y = df[target_col]
    X = df.drop([col for col in df.columns if 'Target' in col], axis=1)
    
    # Chronological Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training on {len(X_train)} samples...")
    
    model = GradientBoostingRegressor(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=5, 
        subsample=0.8, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"Log-Space Model RMSE: {rmse:.6f}, R2 Score: {r2:.4f}")
    
    joblib.dump(model, 'btc_log_model.pkl')
    
    # --- UN-NORMALIZATION (LOG -> REAL PRICE) ---
    scaler = joblib.load('crypto_scaler.pkl')
    
    latest_today = X_test.iloc[-1:]
    pred_scaled = model.predict(latest_today)[0]
    actual_scaled = y_test.iloc[-1]
    
    def to_real_price(val_scaled):
        # 1. Inverse MinMaxScaler
        dummy = np.zeros((1, 10))
        dummy[0, 1] = val_scaled
        log_price = scaler.inverse_transform(dummy)[0, 1]
        
        # 2. Reverse Log (exponential)
        return np.expm1(log_price)

    print("\n--- BTC Price Prediction for Tomorrow ---")
    predicted_price = to_real_price(pred_scaled)
    actual_price = to_real_price(actual_scaled)
    
    print(f"Predicted Price: ${predicted_price:,.2f}")
    print(f"Actual Price:    ${actual_price:,.2f}")
    
    accuracy_pct = (1 - abs(predicted_price - actual_price) / actual_price) * 100
    print(f"Accuracy:        {accuracy_pct:.2f}%")

if __name__ == "__main__":
    train_log_model()
