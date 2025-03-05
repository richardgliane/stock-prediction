import os
import streamlit as st
import plotly.graph_objs as go
from data import fetch_stock_data, preprocess_data, scale_data, create_sequences
from model import LSTMModel, train_model, predict_future
import torch
import pandas as pd

os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"

st.title("Stock Price Prediction Dashboard")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
days_to_predict = st.slider("Days to Predict", 1, 30, 5)

if st.button("Generate Prediction"):
    # Fetch and preprocess data
    df = fetch_stock_data(ticker)
    st.write("Historical Data", df.tail())

    # Plot historical data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close Price"))
    fig.update_layout(title=f"{ticker} Historical Prices", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # Train or load model
    try:
        model = LSTMModel()
        model.load_state_dict(torch.load(f"{ticker}_lstm.pth"))
    except:
        st.write("Training new model...")
        model, scaler = train_model(ticker)
    
    # Predict
    scaled_data, scaler = scale_data(df)
    last_sequence = scaled_data[-60:, 0]
    preds = predict_future(model, scaler, last_sequence, days_to_predict)

    # Plot predictions
    future_dates = pd.date_range(start=df.index[-1], periods=days_to_predict+1, freq='B')[1:]
    fig.add_trace(go.Scatter(x=future_dates, y=preds.flatten(), name="Predictions", line=dict(color='red')))
    st.plotly_chart(fig)

#if __name__ == "__main__":
#    asyncio.run(st._run_session())