import torch
import torch.nn as nn
import numpy as np
from data import fetch_stock_data, scale_data, create_sequences

# Fix for Streamlit-PyTorch compatibility
torch.classes.__path__ = []  # Added by user to resolve RuntimeError

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2):
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
    def __init__(self, input_size=5, hidden_size=50, num_layers=2):
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

class AttentionGRU(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, dropout=0.2):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        gru_out, _ = self.gru(x, h0)
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attention_weights * gru_out, dim=1)
        out = self.fc(context)
        return out

def train_model(ticker, model_type="LSTM", seq_length=60, epochs=20, hidden_size=50, num_layers=2, learning_rate=0.001):
    # Fetch and preprocess data with multiple features
    df = fetch_stock_data(ticker)
    if df.empty:
        raise ValueError(f"Invalid ticker: {ticker}. No data available.")
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaled_data, scaler = scale_data(features)
    X, y = create_sequences(scaled_data, seq_length, multi_feature=True)

    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze(-1)

    # Split into train/test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize model
    if model_type == "LSTM":
        model = LSTMModel(input_size=5, hidden_size=hidden_size, num_layers=num_layers)
    elif model_type == "GRU":
        model = GRUModel(input_size=5, hidden_size=hidden_size, num_layers=num_layers)
    elif model_type == "AttentionGRU":
        model = AttentionGRU(input_size=5, hidden_size=hidden_size, num_layers=num_layers)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Track loss history
    loss_history = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        loss_history.append(loss_value)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_value:.4f}")

    # Save the model
    torch.save(model.state_dict(), f"{ticker}_{model_type.lower()}.pth")
    return model, scaler, loss_history

def predict_future(model, scaler, last_sequence, days=5):
    model.eval()
    future_preds = []
    current_seq = last_sequence.reshape(1, -1, 5)

    for _ in range(days):
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(current_seq)
            pred = model(seq_tensor).item()
            future_preds.append(pred)
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, 3] = pred

    full_scaled = np.zeros((len(future_preds), 5))
    full_scaled[:, 3] = future_preds
    return scaler.inverse_transform(full_scaled)[:, 3]

def create_sequences(data, seq_length=60, multi_feature=False):
    X, y = [], []
    for i in range(seq_length, len(data)):
        if data.shape[1] < 5:
            raise ValueError("Data must have at least 5 features (Open, High, Low, Close, Volume)")
        X.append(data[i-seq_length:i, :])
        y.append(data[i, 3])
    return np.array(X), np.array(y)