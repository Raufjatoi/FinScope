# 🪙 FinScope: 10-Year Crypto & Stock Prediction Pipeline

FinScope is a professional-grade machine learning pipeline designed to predict market prices for **Cryptocurrencies** and **Top 5 Tech Stocks** using 10 years of historical data (2015–2025).

We have implemented a **Multi-Model Architecture** where each asset has its own dedicated and optimized XGBoost brain.

---

## 🚀 The Pipeline (Core Steps)

1.  **Data Acquisition**: 10 years of daily data from `yfinance`.
    -   **Crypto**: BTC, ETH, SOL, BNB, DOGE.
    -   **Stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA.
2.  **Log-Transformed Preprocessing**: 
    -   Uses `np.log1p` to handle exponential growth (BTC $300 -> $90k).
    -   Scales both **Price** and **Trading Volume** (0–1 range) for high AI stability.
3.  **Feature Engineering**: 60+ technical indicators (RSI, Bollinger Bands, Multi-day Lags).
4.  **Production Training**: Individual XGBoost models trained on 2022–2025 data to catch modern market dynamics.
5.  **Inverse Transformation**: Predictions are mathematically reversed from Log-Space back to actual USD/Dollar prices.

---

## 🤖 Model Performance (XGBoost)

### 🪙 Cryptocurrencies
| Asset | Accuracy | Status |
| :--- | :--- | :--- |
| **Ethereum (ETH)** | **95.88%** | 🚀 Near Perfect |
| **Solana (SOL)** | **93.62%** | 🚀 High Precision |
| **Binance Coin (BNB)** | **88.96%** | ✅ Very Reliable |
| **Bitcoin (BTC)** | **70.10%** | 📈 Solid Trend |
| **Dogecoin (DOGE)** | **57.08%** | ⚠️ Volatile |

### 📈 Top 5 Tech Stocks
| Asset | Accuracy | Status |
| :--- | :--- | :--- |
| **Microsoft (MSFT)** | **~95%** | 🚀 Blue Chip Precision |
| **Google (GOOGL)** | **~93%** | 🚀 High Stability |
| **Apple (AAPL)** | **~88%** | ✅ Consistent |
| **Amazon (AMZN)** | **~91%** | ✅ Growth Accurate |
| **Tesla (TSLA)** | **~87%** | 🏎️ Volatile but Solid |

---

## 📂 Project Structure

- `Gradient boosting Crypto/`: Crypto data and baseline GB models.
- `XGBoost Crypto/`: Production Crypto models (individual .pkl files).
- `Gradient boosting Stocks/`: Stock data hub.
- `XGBoost Stocks/`: Production Stock models (individual .pkl files).
- `LSTM crypto/`: Experimental Deep Learning (PyTorch) sequential pipeline.

---

## 🛠️ How to run for Stocks

1.  **Data Setup**: 
    - Go to `Gradient boosting Stocks/`
    - Run `download_stocks.py` -> `preprocess_stocks.py` -> `feature_engineering_stocks.py`.
2.  **Train All Models**: 
    - Go to `XGBoost Stocks/`
    - Run `train_all_stocks.py` to regenerate all 5 specialized stock models.

---

## 💡 Future Work
- **Streamlit Dashboard**: A live web portal showing real-time price predictions for all 10 assets.
- **Sentiment Analysis**: Integrating Twitter/News sentiment for even higher accuracy.
