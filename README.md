# Stock Price Prediction Dashboard

## Overview
This project is an interactive web-based dashboard that predicts future stock prices using historical data and an LSTM (Long Short-Term Memory) neural network. Users can input a stock ticker (e.g., "AAPL" for Apple), visualize historical price trends, training loss, and see predictions for the next few days, along with dynamically fetched company information and logos. Built with Python, it leverages modern data science and machine learning tools to demonstrate skills in time-series analysis, visualization, and deployment.

## Features
- Fetch real-time stock data from Yahoo Finance.
- Predict future stock prices using an LSTM model.
- Interactive visualizations with Plotly, including historical prices, predictions, and training loss.
- User-friendly interface via Streamlit with Enter key trigger for predictions.
- Error handling for invalid tickers.
- Resolved Streamlit-PyTorch runtime compatibility with `torch.classes.__path__ = []`.
- Fixed variable scope issue with `days_to_predict`.
- Removed invalid async workaround causing `AttributeError`.
- Dynamically fetches company name, industry, and sector using `yfinance`.
- Attempts to fetch company logos from Wikipedia (skips if unavailable).
- Option to retrain model or use cached `.pth` file via checkbox.
- Enhanced with custom styling and fade-in animations.

## Demo
Coming Soon: A link to a deployed version will be added once the project is fully tested and deployed.

For now, after running the app locally, you can:
- Enter a stock ticker (e.g., "aapl" or "AAPL") and press Enter to generate predictions (ticker will be capitalized).
- Adjust the prediction horizon (e.g., 5 days) using the slider.
- Toggle the "Retrain Model" checkbox to use a cached model or retrain.
- View historical data, predicted prices, training loss, and company info in a styled interface.

## Tech Stack
- Python: Core language for development.
- yfinance: Fetches historical stock data and company info from Yahoo Finance.
- Pandas: Data manipulation and preprocessing.
- PyTorch: Implements the LSTM model for time-series prediction (with custom fix for Streamlit compatibility).
- Streamlit: Powers the interactive web dashboard with custom styling.
- Plotly: Creates dynamic, interactive visualizations.
- SQLAlchemy (optional): For potential database integration.
- scikit-learn: Provides data scaling (e.g., MinMaxScaler) for preprocessing.
- NumPy: Supports numerical operations and sequence rolling.
- **requests** and **beautifulsoup4**: Used for dynamic logo fetching from Wikipedia (currently non-functional).

## Project Structure
```
stock-price-prediction-dashboard/
├── app.py              # Main Streamlit application
├── model.py            # LSTM model definition and training
├── data.py             # Data fetching and preprocessing
├── requirements.txt    # List of dependencies (to be generated)
├── stockpred_env/      # Virtual environment folder (not tracked in Git)
├── images/             # Folder for screenshots (optional)
└── README.md           # This file
```

## Approach
1. **Data**: Historical stock prices are fetched using yfinance and preprocessed with Pandas and scikit-learn.
2. **Model**: An LSTM model is trained on scaled sequences of closing prices to predict future values, with training loss visualized and caching option.
3. **Visualization**: Plotly charts display historical data, predictions, and training progress with fade-in animations.
4. **UI**: Streamlit ties it all together with an intuitive, styled interface, triggered by Enter key input, including dynamic company info.
   
## Initial Dashboard

![dashboard](https://github.com/richardgliane/stock-prediction/blob/main/images/initial_dashboard.png "Sample Dashboard")

## Latest Dashboard

![dashboard](https://github.com/richardgliane/stock-prediction/blob/main/images/latest_1.png)
![dashboard](https://github.com/richardgliane/stock-prediction/blob/main/images/latest_2.png)
