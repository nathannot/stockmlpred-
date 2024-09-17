import streamlit as st
import yfinance as yf
from darts import TimeSeries
from darts.models import XGBModel, KalmanForecaster, ExponentialSmoothing, RandomForest, LightGBMModel, LinearRegressionModel, AutoARIMA
from darts.dataprocessing.transformers import MissingValuesFiller
import pandas as pd
import datetime
import plotly.graph_objects as go

# App header and description
st.header('Stock Price Forecast using Machine Learning')
st.write('Select from the dropdown box to get forecasted prices!')

# Sidebar for stock selection
st.sidebar.title('Pick the Machine Learning Model or Input Stock Code')

# Initialize session state for stock symbol if it doesn't exist
if 'stock' not in st.session_state:
    st.session_state.stock = 'AAPL'  # Default stock

# Option for predefined popular stocks
predefined_stock = st.selectbox(
    'Choose from the following popular stocks',
    ('AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'CBA.AX', 'BHP.AX', 'CSL.AX', 'WBC.AX', 'NAB.AX'),
    index=('AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'CBA.AX', 'BHP.AX', 'CSL.AX', 'WBC.AX', 'NAB.AX').index(st.session_state.stock)
    if st.session_state.stock in ('AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'CBA.AX', 'BHP.AX', 'CSL.AX', 'WBC.AX', 'NAB.AX')
    else 0
)

# Option for custom stock input
user_stock = st.sidebar.text_input('Enter your own stock code (check Yahoo Finance for code of your desired stock):')
st.sidebar.write('Clear the textbox to pick from dropdown menu again')
# Custom stock entry prioritization
if user_stock:
    st.session_state.stock = user_stock  # Set custom stock if entered
else:
    st.session_state.stock = predefined_stock  # Otherwise, use the predefined stock

# Display which stock is currently selected
stock = st.session_state.stock
st.write(f'Currently selected stock: {stock}')

# Error handling for invalid stock symbols
try:
    st.write('Some machine learning models may take a while')
    
    # Calculate the current date
    current_date = datetime.datetime.today().date()
    # Load stock data from Yahoo Finance
    df = yf.download(stock, start='2019-01-01', end=current_date).reset_index()

    # Transform data using Darts TimeSeries
    data = TimeSeries.from_dataframe(df, time_col='Date', value_cols=['Close'], freq='B')
    fillna = MissingValuesFiller()
    target = fillna.transform(data)

    # Caching the model training process
    @st.cache_resource
    def load_model(stock, model_name,current_date):
        # Initialize the selected model
        if model_name == 'XGBoost (default)':
            model = XGBModel(lags=7, output_chunk_length=4, n_jobs=2, random_state=42)
        elif model_name == 'Random Forest':
            model = RandomForest(lags=7, output_chunk_length=4, n_jobs=2, random_state=42)
        elif model_name == 'LightGBM':
            model = LightGBMModel(lags=7, output_chunk_length=4, n_jobs=2, random_state=42)
        elif model_name == 'Linear Regression':
            model = LinearRegressionModel(lags=7, output_chunk_length=4, n_jobs=2, random_state=42)
        elif model_name == 'Kalman Filter Forecaster':
            model = KalmanForecaster(dim_x=2)
        elif model_name == 'Exponential Smoothing':
            model = ExponentialSmoothing()
        else:
            model = AutoARIMA(start_p=1, start_q=1, start_P=1, start_Q=1, random_state=42)

        # Fit the model on the target stock data
        model.fit(target)
        return model

    # Sidebar to select model
    models = st.sidebar.selectbox(
        'Choose from the following models', 
        ('XGBoost (default)', 'Random Forest', 'LightGBM', 'Linear Regression','Kalman Filter Forecaster','Exponential Smoothing', 'SARIMA')
    )
    st.sidebar.write('Choosing from the different models can be used for research purposes to determine effectiveness of machine learning vs classical models')

    # Load the selected model and train it with the corresponding stock data
    model = load_model(stock, models,current_date)

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
