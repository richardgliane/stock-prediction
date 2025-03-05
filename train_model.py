import torch
import torch.nn as nn
import numpy as np
from data import fetch_stock_data, scale_data, create_sequences

# Fix for Streamlit-PyTorch compatibility
# Added to resolve RuntimeError; Look into this further later
torch.classes.__path__ = []  

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2):  # Input size for 5 features
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2):  # Input size for 5 features
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def train_model(ticker, model_type="LSTM", seq_length=60, epochs=20):
    # Fetch and preprocess data with multiple features
    print(f"Fetching data for ticker: {ticker}")
    df = fetch_stock_data(ticker)
    if df.empty:
        raise ValueError(f"Invalid ticker: {ticker}. No data available.")
    print(f"Data shape: {df.shape}")
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    print(f"Features shape: {features.shape}")
    scaled_data, scaler = scale_data(features)  # Scale all features
    print(f"Scaled data shape: {scaled_data.shape}")
    X, y = create_sequences(scaled_data, seq_length, multi_feature=True)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)  # Shape: (samples, seq_length, features)
    y = torch.FloatTensor(y).unsqueeze(-1)  # Shape: (samples, 1)
    print(f"X tensor shape: {X.shape}, y tensor shape: {y.shape}")

    # Split into train/test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Initialize model
    if model_type == "LSTM":
        model = LSTMModel(input_size=5)  # 5 features: Open, High, Low, Close, Volume
    elif model_type == "GRU":
        model = GRUModel(input_size=5)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"Model type: {model_type}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Track loss history
    loss_history = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)  # Output shape: (batch_size, 1)
        print(f"Epoch {epoch+1}, Output shape: {output.shape}, y_train shape: {y_train.shape}")
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        loss_history.append(loss_value)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_value:.4f}")

    # Save the model
    torch.save(model.state_dict(), f"{ticker}_{model_type.lower()}.pth")
    print(f"Model saved as {ticker}_{model_type.lower()}.pth")
    return model, scaler, loss_history

def predict_future(model, scaler, last_sequence, days=5):
    model.eval()
    future_preds = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(current_seq).reshape(1, -1, 5)  # 5 features
            pred = model(seq_tensor).item()
            future_preds.append(pred)
            # Update the last value (simplified to Close for prediction, index 3)
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, 3] = pred  # Update Close price

    # Inverse transform only the Close price (index 3) from the scaled multi-feature array
    full_scaled = np.zeros((len(future_preds), 5))  # Create a dummy 5-feature array
    full_scaled[:, 3] = future_preds  # Set Close prices
    return scaler.inverse_transform(full_scaled)[:, 3]  # Return only the Close predictions

def create_sequences(data, seq_length=60, multi_feature=False):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :])  # All features
        y.append(data[i, 3])  # Target is Close price (index 3)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Test with a specific ticker
    ticker = "AAPL"
    for model_type in ["LSTM", "GRU"]:
        print(f"\nTraining {model_type} model for {ticker}...")
        try:
            model, scaler, loss_history = train_model(ticker, model_type=model_type, epochs=5)  # Reduced epochs for quick test
            print(f"Training completed for {model_type}. Final loss history: {loss_history}")
        except Exception as e:
            print(f"Error during training {model_type}: {e}")