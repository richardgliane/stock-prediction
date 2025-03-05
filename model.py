
import torch
import torch.nn as nn
import numpy as np
from data import fetch_stock_data, scale_data, create_sequences

#torch.classes.__path__ = [] # add this line to manually set it to empty. 

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
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
        
# Train model
def train_model(ticker, seq_length=60, epochs=20):
    # Fetch and preprocess data
    df = fetch_stock_data(ticker)
    scaled_data, scaler = scale_data(df)
    X, y = create_sequences(scaled_data, seq_length)

    # Convert to PyTorch tensors
    X = torch.FloatTensor(X).reshape(-1, seq_length, 1)
    y = torch.FloatTensor(y)

    # Split into train/test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize model
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), f"{ticker}_lstm.pth")
    return model, scaler        

# Model predictor    
def predict_future(model, scaler, last_sequence, days=5):
    model.eval()
    future_preds = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(current_seq).reshape(1, -1, 1)
            pred = model(seq_tensor).item()
            future_preds.append(pred)
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = pred

    return scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))    