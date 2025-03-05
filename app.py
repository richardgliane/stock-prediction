import os
import streamlit as st
import plotly.graph_objs as go
from data import fetch_stock_data, preprocess_data, scale_data, create_sequences
from model import LSTMModel, train_model, predict_future
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

# User input with Enter key trigger (capitalized)
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL", key="ticker_input").upper()

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
                # Clean company name for Wikipedia URL
                wiki_name = company_name.replace(' ', '_').replace('&', '%26').replace('.', '')
                response = requests.get(f"https://en.wikipedia.org/wiki/{wiki_name}", timeout=5)
                soup = BeautifulSoup(response.text, 'html.parser')
                # Look for logo in infobox or first image
                logo_img = soup.find('img', attrs={'src': lambda x: x and ('logo' in x.lower() or 'svg' in x.lower())})
                if not logo_url:
                    logo_img = soup.find('a', class_='image').find('img') if soup.find('a', class_='image') else None
                    if logo_img and 'src' in logo_img.attrs:
                        src = logo_img['src']
                        logo_url = f"https:{src}" if src.startswith('//') else src
                        if not logo_url.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
                            logo_url = None
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
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close Price"))
            fig.update_layout(title=f"{ticker} Historical Prices", xaxis_title="Date", yaxis_title="Price")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            # Train or load model based on checkbox
            if retrain_model:
                st.write("Training new model...")
                model, scaler, loss_history = train_model(ticker)
                # Plot training loss
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=list(range(1, len(loss_history) + 1)), y=loss_history, mode='lines+markers', name="Training Loss"))
                fig_loss.update_layout(title="Training Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_loss)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                try:
                    model = LSTMModel()
                    model.load_state_dict(torch.load(f"{ticker}_lstm.pth"))
                    st.write("Using cached model...")
                except FileNotFoundError:
                    st.write("No cached model found. Training new model...")
                    model, scaler, loss_history = train_model(ticker)
                    # Plot training loss
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=list(range(1, len(loss_history) + 1)), y=loss_history, mode='lines+markers', name="Training Loss"))
                    fig_loss.update_layout(title="Training Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(fig_loss)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Predict
            scaled_data, scaler = scale_data(df)
            last_sequence = scaled_data[-60:, 0]
            preds = predict_future(model, scaler, last_sequence, days_to_predict)

            # Plot predictions
            future_dates = pd.date_range(start=df.index[-1], periods=days_to_predict + 1, freq='B')[1:]
            fig.add_trace(go.Scatter(x=future_dates, y=preds.flatten(), name="Predictions", line=dict(color='red')))
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")