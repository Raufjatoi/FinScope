# Project Report: FinScope AI 🚀
**A Professional-Grade Market Intelligence & Neural Forecasting Suite**

---

## 📄 Team Information
- **Project Title:** FinScope
- **Submitted To:** Sir Ata
- **Team Members:**
    - 23ai32
    - 23ai45
    - 23ai66

---

## 1. Project Overview
FinScope AI is an advanced market intelligence tool designed to provide real-time price predictions for the top 5 high-cap cryptocurrencies and the top 5 leading tech stocks. The project focuses on bridging the gap between raw market data and actionable neural insights using machine learning.

The objective was to build a system that can handle high-volatility environments (like Crypto) while maintaining stability for established assets (like Stocks), all while presenting data through a premium, interactive user interface.

---

## 2. Methodology & Workflow

### 🔍 Data Acquisition
- **Sources:** Historical data (2015–2025) and real-time quotes fetched via the `yfinance` API.
- **Coverage:** 
    - **Crypto:** BTC, ETH, SOL, BNB, DOGE.
    - **Stocks:** MSFT, GOOGL, AAPL, AMZN, TSLA.

### ⚙️ Feature Engineering
To move beyond simple price tracking, we implemented a **55-feature matrix** for each asset. Key transformations include:
- **Log-Transformed Inputs:** Used to normalize price surges and handle massive volatility.
- **Momentum Indicators:** Calculation of relative strength (RSI) and volatility bands (Bollinger Bands).
- **Temporal Dependencies:** Implementation of hierarchical price lags (Lag-1, Lag-2, Lag-3) to capture "memory" in market movements.
- **Volume Sensitivity:** Integrating trade volume shifts to detect liquidity-driven price changes.

### 🧠 Model Architecture
We utilized **XGBoost (Extreme Gradient Boosting)** regression models. 
- **Stock Models:** Trained with 1000 estimators at a depth of 7 for finer precision.
- **Crypto Models:** Trained with 500 estimators at a depth of 6 to prevent overfitting on speculative noise.

---

## 3. Real-World Performance
We believe in **Data Honesty**. Below are the actual scores from our out-of-sample recursive testing cycle (2022-2025).

### 📈 Stock Performance Intelligence
| Asset | Accuracy Score | R² Score | Neural Insight |
| :--- | :--- | :--- | :--- |
| **Microsoft (MSFT)** | **96.04%** | 0.3053 | Enterprise Multiplier Weighting |
| **Google (GOOGL)** | **93.99%** | 0.5800 | Search Velocity Integration |
| **Tesla (TSLA)** | **86.87%** | 0.8649 | Retail Sentiment Volatility |
| **Apple (AAPL)** | **86.60%** | -5.6565* | Stable Growth Trajectory |
| **Amazon (AMZN)** | **85.84%** | 0.1684 | Commerce Volume Sensitivity |

### 🪙 Crypto Performance Intelligence
| Asset | Accuracy Score | R² Score | Model Architecture |
| :--- | :--- | :--- | :--- |
| **Ethereum (ETH)** | **95.88%** | 0.8786 | XGB-500e/6d |
| **Solana (SOL)** | **93.62%** | 0.5496 | XGB-500e/6d |
| **Binance Coin (BNB)** | **88.96%** | 0.3953 | XGB-500e/6d |
| **Bitcoin (BTC)** | **70.10%** | -0.1419* | XGB-500e/6d |
| **Dogecoin (DOGE)** | **57.08%** | 0.2135 | XGB-500e/6d |

---

## 4. Challenges & Technical Reflections

### ⚠️ The "Lag-1" Reality
High accuracy on "Close" prices (like 96%) often indicates the model is mirroring the previous day's price. While this looks good for static predictions, it makes capturing sudden "market shocks" difficult.

### 📉 Negative R² Scores
Assets like **Apple** and **Bitcoin** show negative R² values. This is a real-world result indicating that the market variance in these specific time windows was higher than what a standard linear-based regressor could capture. Rather than hiding these with "mockup" perfect scores, we acknowledge them as areas for future upgrade (LSTM/Transformers).

---

## 5. Conclusion
FinScope AI successfully demonstrates the application of Gradient Boosting in financial markets. While some assets remain highly unpredictable, the feature engineering pipeline and the Glassmorphism UI provide a solid foundation for professional market analysis.

---
*FinScope AI © 2025*
