import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_crypto_stats():
    DATA_DIR = os.path.join(BASE_DIR, "Gradient boosting Crypto")
    MODEL_DIR = os.path.join(BASE_DIR, "XGBoost Crypto")
    feat_path = os.path.join(DATA_DIR, "crypto_features_advanced.csv")
    scaler_path = os.path.join(DATA_DIR, "crypto_scaler.pkl")
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    df = df[df.index >= '2022-01-01']
    scaler = joblib.load(scaler_path)
    tickers = {'BNB-USD': 0, 'BTC-USD': 1, 'DOGE-USD': 2, 'ETH-USD': 3, 'SOL-USD': 4}
    for ticker, idx in tickers.items():
        try:
            model_path = os.path.join(MODEL_DIR, f"{ticker.lower().split('-')[0]}_xgboost_model.pkl")
            if not os.path.exists(model_path): 
                print(f"CRYPTO|{ticker}|NOT_FOUND")
                continue
            model = joblib.load(model_path)
            y = df[f'{ticker}_Target']
            X = df.drop([col for col in df.columns if 'Target' in col], axis=1)
            split_idx = int(len(df) * 0.85)
            X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            dummy_pred = np.zeros((1, 10))
            dummy_pred[0, idx] = preds[-1]
            dummy_actual = np.zeros((1, 10))
            dummy_actual[0, idx] = y_test.iloc[-1]
            price_pred = np.expm1(scaler.inverse_transform(dummy_pred)[0, idx])
            price_actual = np.expm1(scaler.inverse_transform(dummy_actual)[0, idx])
            acc = (1 - abs(price_pred - price_actual) / price_actual) * 100
            print(f"CRYPTO|{ticker}|{r2:.4f}|{acc:.2f}")
        except Exception as e:
            print(f"CRYPTO|{ticker}|ERROR|{str(e)}")

def get_stock_stats():
    DATA_DIR = os.path.join(BASE_DIR, "Gradient boosting Stocks")
    MODEL_DIR = os.path.join(BASE_DIR, "XGBoost Stocks")
    feat_path = os.path.join(DATA_DIR, "stocks_features_advanced.csv")
    scaler_path = os.path.join(DATA_DIR, "stocks_scaler.pkl")
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    df = df[df.index >= '2022-01-01']
    scaler = joblib.load(scaler_path)
    tickers = {'AAPL': 0, 'AMZN': 1, 'GOOGL': 2, 'MSFT': 3, 'TSLA': 4}
    for ticker, idx in tickers.items():
        try:
            model_path = os.path.join(MODEL_DIR, f"{ticker.lower()}_xgboost_model.pkl")
            if not os.path.exists(model_path): 
                print(f"STOCK|{ticker}|NOT_FOUND")
                continue
            model = joblib.load(model_path)
            y = df[f'{ticker}_Target']
            X = df.drop([col for col in df.columns if 'Target' in col], axis=1)
            split_idx = int(len(df) * 0.85)
            X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            dummy_pred = np.zeros((1, 10))
            dummy_pred[0, idx] = preds[-1]
            dummy_actual = np.zeros((1, 10))
            dummy_actual[0, idx] = y_test.iloc[-1]
            price_pred = np.expm1(scaler.inverse_transform(dummy_pred)[0, idx])
            price_actual = np.expm1(scaler.inverse_transform(dummy_actual)[0, idx])
            acc = (1 - abs(price_pred - price_actual) / price_actual) * 100
            print(f"STOCK|{ticker}|{r2:.4f}|{acc:.2f}")
        except Exception as e:
            print(f"STOCK|{ticker}|ERROR|{str(e)}")

get_crypto_stats()
get_stock_stats()
