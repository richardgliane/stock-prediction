
import yfinance as yf
import pandas as pd

# Use yfinance to download stock data
def fetch_stock_data(ticker, start_date="2020-01-01", end_date="2025-03-05"):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def preprocess_data(df):
    # Use 'Close' price for prediction
    df = df[['Close']].dropna()
    return df
    

# Normalize the data for the LSTM model    
from sklearn.preprocessing import MinMaxScaler

def scale_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    return scaled_data, scaler
    

# LSTM needs sequences of past data to predict future values    
import numpy as np

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)    