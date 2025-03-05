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

## Updated Dashboard
Example of the dashboard showing historical prices, predictions, statically plotted training loss, and company info.
![dashboard](https://github.com/richardgliane/stock-prediction/blob/main/images/latest_1.png)
![dashboard](https://github.com/richardgliane/stock-prediction/blob/main/images/latest_2.png)


## Findings
The price predictions don’t emulate real-world price movements effectively. This is a common challenge in stock price prediction, as financial markets are influenced by complex, non-linear factors including macroeconomic indicators, sentiment, volume, and unexpected events, which simple LSTM models may not capture well. The current LSTM model, while a solid starting point for time-series forecasting, has limitations such as:

**Over-simplification**: It relies solely on historical closing prices, ignoring other features like volume, open/high/low prices, or external data.

**Stationarity Assumption**: LSTMs assume the time series is somewhat stationary, but stock prices are often non-stationary with trends and volatility clusters.

**Short Memory**: The 60-day sequence length might not capture long-term dependencies or seasonal patterns.

**Lack of Context**: It doesn’t account for market sentiment, news, or technical indicators (e.g., RSI, MACD).

To improve prediction accuracy and better emulate real-world price movements, we can explore better models and techniques. Below, I’ll propose enhancements and an additional model option, then update the code to implement one of these as a selectable alternative.

## Analysis and Proposed Improvements
### Limitations of the Current LSTM
- The model predicts a linear continuation based on past trends, which doesn’t reflect the volatility or sudden shifts in stock prices.
- The training data (80% train/20% test split on scaled closing prices) might overfit to noise rather than meaningful patterns.
- The loss function (MSE) penalizes all errors equally, which might not align with the need to capture directional changes.

### Better Models and Techniques

**Enhanced LSTM with Additional Features**:
* Incorporate features like volume, open/high/low prices, moving averages, and technical indicators (e.g., RSI, Bollinger Bands).
* Use a multi-input LSTM to process these features, potentially improving pattern recognition.

**GRU (Gated Recurrent Unit)**:
* A simpler alternative to LSTM with fewer parameters, which can be faster to train and still capture sequential dependencies.
* Might perform better with noisy data like stock prices.

**Transformer-Based Models (e.g., Time Series Transformer)**:
* Transformers excel at capturing long-range dependencies and can handle multivariate time series.
* Requires more data and computational resources but offers state-of-the-art performance for time-series forecasting.

**Hybrid Models (LSTM + Attention)**:
* Add an attention mechanism to the LSTM to focus on important time steps, improving the model’s ability to weigh recent vs. distant data.
* This can better emulate sudden price movements driven by specific events.

**Ensemble Methods**:
* Combine predictions from LSTM, GRU, and a statistical model (e.g., ARIMA) to leverage diverse strengths.
* Reduces overfitting and improves robustness.

**External Data Integration**:
* Include sentiment analysis from news or social media (e.g., Twitter), macroeconomic indicators (e.g., interest rates), or event data.
* Use a model like BERT for sentiment, combined with LSTM for price data.

**Volatility Modeling**:
* Add a GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model or a stochastic volatility model to capture price volatility, which is critical for realistic predictions.

### Recommended Approach
Given the current codebase and constraints (simplicity, existing dependencies), let’s:
- Enhance the existing LSTM with additional features (open, high, low, volume) to improve input diversity.
- Add a GRU model as an alternative option, selectable via a radio button, to compare performance.
- Keep the implementation manageable by avoiding complex transformers or external data for now, which can be added later.

## Dashboard with multiple models (LTSM, GRU)
Improved performanced can be seen below with the GRU model
![dashboard](https://github.com/richardgliane/stock-prediction/blob/main/images/tsla_gru_1.png)
![dashboard](https://github.com/richardgliane/stock-prediction/blob/main/images/tsla_gru_2.png)
