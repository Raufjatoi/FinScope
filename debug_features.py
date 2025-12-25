import joblib
import xgboost as xgb
import os

model_path = os.path.join("XGBoost Stocks", "aapl_xgboost_model.pkl")
model = joblib.load(model_path)
names = model.get_booster().feature_names
print(f"Total features: {len(names)}")
for n in names:
    print(f"'{n}'")
