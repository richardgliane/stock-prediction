import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data(ticker, start_date="2020-01-01", end_date=None):
    stock = yf.Ticker(ticker)
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    df = stock.history(start=start_date, end=end_date)
    if df.empty:
        return df
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)  # Ensure this returns a 2D array (n, 5)
    return scaled_data, scaler

def create_sequences(data, seq_length=60, multi_feature=False):
    X, y = [], []
    for i in range(seq_length, len(data)):
        # Ensure data is 2D and has at least 5 columns
        if data.shape[1] < 5:
            raise ValueError("Data must have at least 5 features (Open, High, Low, Close, Volume)")
        X.append(data[i-seq_length:i, :])  # All 5 features
        y.append(data[i, 3])  # Target is Close price (index 3)
    return np.array(X), np.array(y)

def preprocess_data(df):
    return df