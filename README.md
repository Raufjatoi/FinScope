# 🪙 FinScope: 10-Year Crypto Prediction Pipeline

FinScope is a professional-grade machine learning pipeline designed to predict Cryptocurrency prices using 10 years of historical data (2015–2025). 

We have implemented a **Multi-Model Architecture** where each cryptocurrency has its own dedicated and optimized XGBoost brain.

---

## 🚀 The Pipeline (Comprehensive)

1.  **Data Acquisition**: 10 years of daily data for **BTC, ETH, SOL, BNB, and DOGE**.
2.  **Log-Transformed Preprocessing**: 
    -   Uses `np.log1p` to handle 10 years of exponential growth.
    -   Scales both **Price** and **Trading Volume** (0–1 range).
3.  **Feature Engineering**: 60+ indicators including RSI, Bollinger Bands, and multi-day lags.
4.  **Inverse Transformation**: Predictions are mathematically reversed from Log-Space back to actual USD prices.

---

## 🤖 Model Performance (XGBoost Refined)

We achieved extreme accuracy by focusing our XGBoost models on the 2022–2025 market period. Each coin is saved as a separate `.pkl` model.

| Cryptocurrency | Accuracy | Status |
| :--- | :--- | :--- |
| **Ethereum (ETH)** | **95.88%** | 🚀 Near Perfect |
| **Solana (SOL)** | **93.62%** | 🚀 High Precision |
| **Binance Coin (BNB)** | **88.96%** | ✅ Very Reliable |
| **Bitcoin (BTC)** | **70.10%** | 📈 Solid Trend |
| **Dogecoin (DOGE)** | **57.08%** | ⚠️ Volatile |

---

## 📂 Project Structure

- `Gradient boosting Crypto/`: Baseline pipeline and 10-year training data.
- `XGBoost Crypto/`: **Production Hub**. Contains individual model files:
  - `btc_xgboost_model.pkl`
  - `eth_xgboost_model.pkl`
  - `sol_xgboost_model.pkl`
  - `bnb_xgboost_model.pkl`
  - `doge_xgboost_model.pkl`
- `LSTM crypto/`: Deep Learning (PyTorch) experimental implementation.

---

## 🛠️ How to Run

1.  **Data Setup**: 
    - Go to `Gradient boosting Crypto/`
    - Run `download_crypto.py` -> `preprocess_crypto.py` -> `feature_engineering_crypto.py`.
2.  **Train All Models**: 
    - Go to `XGBoost Crypto/`
    - Run `train_all_cryptos.py` to regenerate all 5 specialized models.

---

## 💡 Future Work
- **Streamlit Integration**: A live web dashboard to display these real-time predictions.
- **Stock Market Pipeline**: Expanding this architecture to predict 10 years of Stock data.
