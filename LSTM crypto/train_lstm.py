import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score

# Path helpers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Gradient boosting Crypto")

class CryptoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CryptoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Take the last time step
        return out

def prepare_sequences(data, target, look_back=10):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(target[i + look_back])
    return np.array(X), np.array(y)

def train_lstm_pytorch():
    print("Loading data for PyTorch LSTM...")
    feat_path = os.path.join(DATA_DIR, "crypto_features_advanced.csv")
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    df = df[df.index >= '2020-01-01']
    
    X_raw = df.drop([col for col in df.columns if 'Target' in col], axis=1).values
    y_raw = df['BTC-USD_Close'].values
    
    look_back = 10
    X_seq, y_seq = prepare_sequences(X_raw, y_raw, look_back)
    
    # Split
    split_idx = int(len(X_seq) * 0.85)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    
    # Model Setup
    input_size = X_train.shape[2]
    hidden_size = 64
    num_layers = 2
    output_size = 1
    
    model = CryptoLSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting Training (20 epochs)...")
    model.train()
    for epoch in range(20):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/20], Loss: {loss.item():.6f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).numpy()
        
    rmse = np.sqrt(mean_squared_error(y_test, preds_scaled))
    r2 = r2_score(y_test, preds_scaled)
    print(f"LSTM Model RMSE (Log Space): {rmse:.6f}, R2 Score: {r2:.4f}")
    
    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'btc_lstm_weights.pth'))
    
    # --- UN-NORMALIZATION ---
    scaler_path = os.path.join(DATA_DIR, "crypto_scaler.pkl")
    scaler = joblib.load(scaler_path)
    
    last_pred_scaled = preds_scaled[-1][0]
    last_actual_scaled = y_test[-1]
    
    def to_real_price(val_scaled):
        dummy = np.zeros((1, 10))
        dummy[0, 1] = val_scaled
        log_price = scaler.inverse_transform(dummy)[0, 1]
        return np.expm1(log_price)

    predicted_price = to_real_price(last_pred_scaled)
    actual_price = to_real_price(last_actual_scaled)

    print("\n--- PyTorch LSTM BTC Prediction ---")
    print(f"Predicted Price: ${predicted_price:,.2f}")
    print(f"Actual Price:    ${actual_price:,.2f}")
    accuracy_pct = (1 - abs(predicted_price - actual_price) / actual_price) * 100
    print(f"Accuracy:        {accuracy_pct:.2f}%")

if __name__ == "__main__":
    train_lstm_pytorch()
