import streamlit as st
import yfinance as yf
from darts import TimeSeries
from darts.models import XGBModel, RandomForest, LightGBMModel, LinearRegressionModel, AutoARIMA
from darts.dataprocessing.transformers import MissingValuesFiller
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go

st.header('Stock Price Forecast using Machine Learning')
st.write('Select from the dropdown box or enter your own stock code to get forecasted prices!')

st.sidebar.title('Pick the Machine Learning Model or Input Stock Code')
# Option for predefined stocks
predefined_stock = st.selectbox('Choose from the following popular stocks', 
    ('AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'CBA.AX', 'BHP.AX', 'CSL.AX', 'WBC.AX', 'NAB.AX'))

# Option for custom stock input
user_stock = st.sidebar.text_input('Enter your own stock code if you know it (e.g., AAPL for Apple, TSLA for Tesla):')
st.sidebar.write('Note: Does not work for crypto, only business day financial instruments.')
# If a user enters a stock code, prioritize that over the dropdown selection
if st.sidebar.button('Use Custom Stock') and user_stock:
    stock = user_stock
else:
    stock = predefined_stock

st.write('Some machine learning models may take a while')

# Get the current date
current_date = datetime.datetime.today().date()

# Error handling for invalid stock symbols
try:
    # Load stock data from Yahoo Finance
    df = yf.download(stock, start='2019-01-01', end=current_date).reset_index()

    # Transform data using Darts TimeSeries
    data = TimeSeries.from_dataframe(df, time_col='Date', value_cols=['Close'], freq='B')
    fillna = MissingValuesFiller()
    target = fillna.transform(data)

    # Sidebar to select model
    st.sidebar.write('For advanced users, pick from the following ML models or SARIMA')

    # Caching the model training process with stock symbol as a parameter
    @st.cache_resource
    def load_model(stock, model_name):
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
        
        # Fit the model (caching fitted model)
        model.fit(target)
        return model

    # Sidebar to select model
    models = st.sidebar.selectbox(
        'Choose from the following models', 
        ('XGBoost (default)','Random Forest',  'LightGBM', 'Linear Regression', 'SARIMA'))

    # Load the selected model and train
    model = load_model(stock, models)

    # User inputs for past data and forecast
    days = st.slider('Pick how many past days to view from last year', min_value=1, max_value=365, value=30)
    forecast = st.slider('Pick Forecast Period (smaller will be more accurate)', min_value=1, max_value=60, value=7)

    # Predict future prices
    predx = model.predict(forecast)

    # Plot past data and forecast
    st.write(f"Chart of {stock}'s past {days}-days price and {forecast}-day forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=target[-days:].pd_dataframe().index, y=target[-days:].pd_dataframe().Close, name=f'Past {days} days'))
    fig.add_trace(go.Scatter(x=predx.pd_dataframe().index, y=predx.pd_dataframe().Close, name=f'{forecast} day forecast'))
    fig.update_layout(hovermode='x', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
    st.plotly_chart(fig)

    # Display forecast table
    st.write(f"Table of {stock}'s {forecast}-day Forecasted Values")
    table = predx.pd_dataframe().round(2)
    st.write(table)

    st.write('Note this app is for educational purposes, you can compare how these stocks fair to the forecasted values')
    st.write('This app is NOT to be used to make investment decisions')

except Exception as e:
    st.error(f"Error: Unable to retrieve data for the stock symbol '{stock}'. Please check the stock symbol and try again.")
