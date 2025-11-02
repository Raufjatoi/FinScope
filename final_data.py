import pandas as pd

scaled_df = pd.read_csv("crypto_scaled.csv", index_col=0)

def make_features(df, coin):
    temp = df[[coin]].copy()
    
    # Lag features (previous days)
    for i in range(1, 4):
        temp[f"{coin}_Prev_{i}"] = temp[coin].shift(i)
    
    # Moving averages
    temp[f"{coin}_MA_5"] = temp[coin].rolling(window=5).mean()
    temp[f"{coin}_MA_10"] = temp[coin].rolling(window=10).mean()
    
    # Price momentum (% change)
    temp[f"{coin}_Change"] = temp[coin].pct_change()
    
    # Short-term volatility (std)
    temp[f"{coin}_Volatility"] = temp[coin].rolling(window=5).std()
    
    temp = temp.dropna()
    return temp

# --- Apply to all 5 cryptos ---
features_list = []
for coin in scaled_df.columns:
    features = make_features(scaled_df, coin)
    features_list.append(features)

# --- Merge all ---
final_df = pd.concat(features_list, axis=1).dropna()

# --- Save final data ---
final_df.to_csv("crypto_final.csv")

print("✅ Final feature-rich crypto dataset saved as 'crypto_final.csv'")
print("📊 Shape:", final_df.shape)
print("\n📄 Preview:")
print(final_df.head())
