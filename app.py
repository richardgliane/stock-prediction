import os
import streamlit as st
import plotly.graph_objs as go
from data import fetch_stock_data, scale_data, create_sequences
from model import LSTMModel, GRUModel, AttentionGRU, train_model, predict_future
import torch
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# Custom CSS for styling and animations
st.markdown(
    """
    <style>
    .title {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
        text-align: center;
        padding: 10px;
        background-color: #ecf0f1;
        border-radius: 5px;
    }
    .company-info {
        color: #34495e;
        font-size: 16px;
        padding: 10px;
        background-color: #f9f9f9;
        border-left: 4px solid #3498db;
        margin-bottom: 20px;
        animation: fadeIn 1s ease-in;
    }
    .chart-container {
        animation: fadeIn 1.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"

st.markdown('<h1 class="title">Stock Price Prediction Dashboard</h1>', unsafe_allow_html=True)

# Initialize session state for ticker if not present
if 'ticker' not in st.session_state:
    st.session_state.ticker = "AAPL"

# User input with real-time capitalization
def update_ticker():
    st.session_state.ticker = st.session_state.ticker_input.upper()

ticker = st.text_input("Enter Stock Ticker", value=st.session_state.ticker, key="ticker_input", on_change=update_ticker)

# Model selection with full names in brackets
model_type_full = st.radio("Select Model", ["LSTM (Long Short-Term Memory)", "GRU (Gated Recurrent Unit)", "AttentionGRU (Attention-based Gated Recurrent Unit)"], index=0)
model_type = model_type_full.split(' (')[0]  # Extract short name (e.g., "AttentionGRU")

# Hyperparameter tuning sliders
epochs = st.slider("Number of Epochs", min_value=5, max_value=100, value=20, step=5)
hidden_size = st.slider("Hidden Size", min_value=20, max_value=200, value=50, step=10)
num_layers = st.slider("Number of Layers", min_value=1, max_value=5, value=2, step=1)
learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")

# Set days_to_predict slider (default to 5)
days_to_predict = st.slider("Days to Predict", 1, 30, 5, key="days_slider")

# Checkbox to retrain model
retrain_model = st.checkbox("Retrain Model (uncheck to use cached model)", value=True)

# Trigger prediction on input change (Enter key)
if ticker:
    try:
        # Fetch and preprocess data
        df = fetch_stock_data(ticker)
        if df.empty:
            st.error(f"Invalid ticker: {ticker}. Please enter a valid stock ticker.")
        else:
            # Fetch company info dynamically
            stock = yf.Ticker(ticker)
            company_name = stock.info.get("longName", "Unknown Company")
            industry = stock.info.get("industry", "No industry information available")
            sector = stock.info.get("sector", "No sector information available")
            info = f"{company_name} - Industry: {industry}, Sector: {sector}"

            # Dynamically fetch logo from Wikipedia (currently non-functional, debug included)
            logo_url = None
            try:
                wiki_name = company_name.replace(' ', '_').replace('&', '%26').replace('.', '')
                response = requests.get(f"https://en.wikipedia.org/wiki/{wiki_name}", timeout=5)
                soup = BeautifulSoup(response.text, 'html.parser')
                logo_img = soup.find('img', attrs={'src': lambda x: x and ('logo' in x.lower() or 'svg' in x.lower())})
                if not logo_img:
                    logo_img = soup.find('table', class_='infobox').find('img') if soup.find('table', class_='infobox') else None
                if logo_img and 'src' in logo_img.attrs:
                    src = logo_img['src']
                    logo_url = f"https:{src}" if src.startswith('//') else src
                    if not logo_url.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
                        logo_url = None
            except Exception as e:
                st.write(f"Logo fetch failed: {e}")  # Debug output

            # Display company info and logo if available
            st.markdown(f'<div class="company-info">Company: {info}</div>', unsafe_allow_html=True)
            if logo_url:
                st.markdown(f'<img src="{logo_url}" alt="Company Logo" width="100" style="display:block; margin-left:auto; margin-right:auto;">', unsafe_allow_html=True)

            st.write("Historical Data", df.tail())

            # Plot historical data
            fig_historical = go.Figure()
            fig_historical.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close Price"))
            fig_historical.update_layout(title=f"{ticker} Historical Prices", xaxis_title="Date", yaxis_title="Price")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_historical)
            st.markdown('</div>', unsafe_allow_html=True)

            # Train or load model based on checkbox
            model = None
            if retrain_model:
                st.write(f"Training new {model_type_full} model with epochs={epochs}, hidden_size={hidden_size}, layers={num_layers}, lr={learning_rate}...")
                with st.spinner("Training in progress..."):
                    model, scaler, loss_history = train_model(ticker, model_type=model_type, epochs=epochs, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate)
                # Plot training loss statically
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=list(range(1, len(loss_history) + 1)), y=loss_history, mode='lines+markers', name="Training Loss"))
                fig_loss.update_layout(title=f"{model_type_full} Training Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_loss)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                try:
                    model = LSTMModel(input_size=5, hidden_size=hidden_size, num_layers=num_layers) if model_type == "LSTM" else \
                            GRUModel(input_size=5, hidden_size=hidden_size, num_layers=num_layers) if model_type == "GRU" else \
                            AttentionGRU(input_size=5, hidden_size=hidden_size, num_layers=num_layers)
                    model.load_state_dict(torch.load(f"{ticker}_{model_type.lower()}.pth"))
                    st.write(f"Using cached {model_type_full} model...")
                except FileNotFoundError:
                    st.write(f"No cached {model_type_full} model found. Training new model with epochs={epochs}, hidden_size={hidden_size}, layers={num_layers}, lr={learning_rate}...")
                    with st.spinner("Training in progress..."):
                        model, scaler, loss_history = train_model(ticker, model_type=model_type, epochs=epochs, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate)
                    # Plot training loss statically
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=list(range(1, len(loss_history) + 1)), y=loss_history, mode='lines+markers', name="Training Loss"))
                    fig_loss.update_layout(title=f"{model_type_full} Training Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(fig_loss)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Predict and plot predictions after training is complete
            if model:
                scaled_data, scaler = scale_data(df[['Open', 'High', 'Low', 'Close', 'Volume']].values)
                last_sequence = scaled_data[-60:, :].reshape(1, -1, 5)
                preds = predict_future(model, scaler, last_sequence, days_to_predict)

                prediction_container = st.container()
                with prediction_container:
                    fig_prediction = go.Figure()
                    fig_prediction.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close Price"))
                    fig_prediction.add_trace(go.Scatter(x=pd.date_range(start=df.index[-1], periods=days_to_predict + 1, freq='B')[1:], y=preds, name=" Predictions", line=dict(color='red')))
                    fig_prediction.update_layout(title=f"{ticker} Historical Prices with {model_type_full} Predictions", xaxis_title="Date", yaxis_title="Price")
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(fig_prediction, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")