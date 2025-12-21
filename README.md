# 🪙 FinScope: 10-Year Crypto Prediction Pipeline

FinScope is a professional-grade machine learning pipeline designed to predict Cryptocurrency prices (specifically **Bitcoin**) using 10 years of historical data (2015–2025). 

I implemented three different AI architectures to find the best way to handle the extreme volatility and growth of the crypto market.

---

## 🚀 The Pipeline (Step-by-Step)

The project follows a rigorous 4-step process to ensure data quality and prediction accuracy:

1.  **Data Acquisition**: Automatically downloads 10 years of daily data for BTC, ETH, SOL, BNB, and DOGE using `yfinance`.
2.  **Preprocessing (The "Secret Sauce")**: 
    -   **Log Transformation**: Because Bitcoin grew from $300 to $90,000, we use natural logs to make the price growth linear and "learnable" for the AI.
    -   **MinMaxScaler**: Shrinks all data to a 0–1 range for faster training.
    -   **Persistent Scaler**: We save the specific "math" used for scaling so we can perfectly convert predictions back to USD.
3.  **Feature Engineering**: Generates 60+ indicators including:
    -   **RSI (Momentum)**: Is the coin overbought?
    -   **Bollinger Bands**: Is the price in a high-volatility zone?
    -   **Volume Lags**: How much money was traded in the last 3 days?
4.  **Inverse Transformation**: Every prediction made as a decimal (e.g., `0.85`) is mathematically converted back into a real-world price (e.g., `$92,643.21`).

---

## 🤖 Model Performance Comparison

We tested three distinct "brains" for this project:

| Model Folder | AI Architecture | Target Data | Best Accuracy | Key Strength |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting** | Decision Trees | 2015–2025 | **~63%** | Very stable baseline for 10-year trends. |
| **XGBoost Crypto** | Optimized Boosting | 2022–2025 | **~70.10%** | **Winner.** Excellent at catching recent spikes. |
| **LSTM Crypto** | Neural Network | Sequential | **~56%** | "Remembers" patterns over time (PyTorch). |

---

## 📂 Project Structure

- `Gradient boosting Crypto/`: The core data and the baseline model.
- `XGBoost Crypto/`: The high-performance refined model focusing on recent 2022-2025 data.
- `LSTM crypto/`: The Deep Learning (PyTorch) implementation for sequential pattern matching.

---

## 🛠️ How to Run

If you want to reproduce these results:

1.  **Generate Data**: Go to `Gradient boosting Crypto/` and run `download_crypto.py`, then `preprocess_crypto.py`, then `feature_engineering_crypto.py`.
2.  **Run Models**:
    -   For the baseline: Run `train_crypto_model.py` in the GB folder.
    -   For the best results: Go to `XGBoost Crypto/` and run `train_xgboost.py`.
    -   For Deep Learning: Go to `LSTM crypto/` and run `train_lstm.py`.

---

## 💡 Why Log Scaling Matters?
If we don't use log scaling, the model thinks the price movements in 2015 ($10 moves) are "zero" compared to 2025 ($2,000 moves). By using **Log Transformation**, we tell the AI that a 10% move is important regardless of whether the price is $1,000 or $100,000. This is how we achieved realistic price predictions!
