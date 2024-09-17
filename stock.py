import streamlit as st
import yfinance as yf
from darts import TimeSeries
from darts.models import XGBModel, RandomForest, LightGBMModel, LinearRegressionModel, AutoARIMA
from darts.dataprocessing.transformers import MissingValuesFiller
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go

# App header and description
st.header('Stock Price Forecast using Machine Learning')
st.write('Select from the dropdown box to get forecasted prices!')

# Sidebar for stock selection
st.sidebar.title('Pick the Machine Learning Model or Input Stock Code')

# Option for predefined popular stocks
predefined_stock = st.selectbox('Choose from the following popular stocks', 
    ('AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'CBA.AX', 'BHP.AX', 'CSL.AX', 'WBC.AX', 'NAB.AX'))

# Option for custom stock input
user_stock = st.sidebar.text_input('Enter your own stock code if you know it (e.g., AAPL for Apple, TSLA for Tesla):')
st.sidebar.write('Note: Does not work for crypto, only business day financial instruments.')

# If user enters a custom stock symbol, prioritize it over predefined stocks
if st.sidebar.button('Use Custom Stock') and user_stock:
    stock = user_stock
else:
    stock = predefined_stock

st.write('Some machine learning models may take a while')

# Error handling for invalid stock symbols
try:
    # Sidebar for machine learning model selection
    st.sidebar.write('For advanced users, pick from the following ML models or SARIMA')
    # Calculate the current date inside the function for daily data update
    current_date = datetime.datetime.today().date()
    # Load stock data from Yahoo Finance with up-to-date current date
    df = yf.download(stock, start='2019-01-01', end=current_date).reset_index()

    # Transform data using Darts TimeSeries
    data = TimeSeries.from_dataframe(df, time_col='Date', value_cols=['Close'], freq='B')
    fillna = MissingValuesFiller()
    target = fillna.transform(data)
    # Caching the model training process with stock symbol as a parameter
    @st.cache_resource
    def load_model(stock, model_name):

        # Initialize the selected model
        if model_name == 'XGBoost (default)':
            model = XGBModel(lags=7, output_chunk_length=4, n_jobs=2, random_state=42)
        elif model_name == 'Random Forest':
            model = RandomForest(lags=7, output_chunk_length=4, n_jobs=2, random_state=42)
        elif model_name == 'LightGBM':
            model = LightGBMModel(lags=7, output_chunk_length=4, n_jobs=2, random_state=42)
        elif model_name == 'Linear Regression':
            model = LinearRegressionModel(lags=7, output_chunk_length=4, n_jobs=2, random_state=42)
        else:
            model = AutoARIMA(start_p=1, start_q=1, start_P=1, start_Q=1, random_state=42)

        # Fit the model on the target stock data
        model.fit(target)
        return model

    # Sidebar to select model
    models = st.sidebar.selectbox(
        'Choose from the following models', 
        ('XGBoost (default)', 'Random Forest', 'LightGBM', 'Linear Regression', 'SARIMA')
    )

    # Load the selected model and train it with the corresponding stock data
    model = load_model(stock, models)

    # User input sliders for number of past days and forecast period
    days = st.slider('Pick how many past days to view from last year', min_value=1, max_value=365, value=30)
    forecast = st.slider('Pick Forecast Period (smaller will be more accurate)', min_value=1, max_value=60, value=7)

    # Predict future prices using the selected model
    predx = model.predict(forecast)

    # Plot the past stock prices and the forecasted prices
    st.subheader(f"Chart of {stock}'s past {days}-days price and {forecast}-day forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=target[-days:].pd_dataframe().index, y=target[-days:].pd_dataframe().Close, name=f'Past {days} days'))
    fig.add_trace(go.Scatter(x=predx.pd_dataframe().index, y=predx.pd_dataframe().Close, name=f'{forecast} day forecast'))
    fig.update_layout(hovermode='x', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
    st.plotly_chart(fig)

    # Display forecast table
    st.write(f"Table of {stock}'s {forecast}-day Forecasted Values")
    table = predx.pd_dataframe().round(2)
    st.write(table)

    # Disclaimer for educational purposes
    st.write('Note this app is for educational purposes, you can compare how these stocks fare to the forecasted values')
    st.write('This app is NOT to be used to make investment decisions')

except Exception as e:
    st.error(f"Error: Unable to retrieve data for the stock symbol '{stock}'. Please check the stock symbol and try again.")
