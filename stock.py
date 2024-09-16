import streamlit as st
import yfinance as yf
from darts import TimeSeries
from darts.models import XGBModel, RandomForest, LightGBMModel, LinearRegressionModel, AutoARIMA
from darts.dataprocessing.transformers import MissingValuesFiller
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import joblib

st.header('Stock Price Forecast using Machine Learning')
st.write('Select from dropdown box which stock to get forecasted prices (pick forecast period below) for these popular US and AUS stocks!')

stock = st.selectbox('Choose from the following stocks', ('AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'CBA.AX', 'BHP.AX', 'CSL.AX', 'WBC.AX', 'NAB.AX'))


current_date = datetime.datetime.today().date()


df = yf.download(stock, start='2019-01-01', end=current_date).reset_index()
    
   
data = TimeSeries.from_dataframe(df, time_col='Date', value_cols=['Close'], freq='B')
fillna = MissingValuesFiller()
target = fillna.transform(data)

st.sidebar.title('Pick Machine Learning Model')
st.sidebar.write('For advanced users, pick from the following ML models or SARIMA')

# Model selection logic
models = st.sidebar.selectbox('Choose from the following models', ('XGBoost (default)', 'Random Forest', 'LightGBM', 'Linear Regression', 'SARIMA'))

if models == 'XGBoost (default)':
    model = XGBModel(lags=7, output_chunk_length=4, n_jobs=-2, random_state=42)
elif models == 'Random Forest':
    model = RandomForest(lags=7, output_chunk_length=4, n_jobs=2, random_state=42)
elif models == 'LightGBM':
    model = LightGBMModel(lags=7, output_chunk_length=4, n_jobs=2, random_state=42)
elif models == 'Linear Regression':
    model = LinearRegressionModel(lags=7, output_chunk_length=4, n_jobs=2, random_state=42)
else:
    model = AutoARIMA(start_p=1, start_q=1, start_P=1, start_Q=1, random_state=42)

model.fit(target)


# Plot past data and forecast
days = st.slider('Pick how many past days to view from last year', min_value=1, max_value=365, value=30)
forecast = st.slider('Pick Forecast Period (smaller will be more accurate)', min_value=1,max_value=60,value=7)
predx = model.predict(forecast)
st.write(f"Chart of {stock}'s past {days}-days price and {forecast}-day forecast")
fig = go.Figure()
fig.add_trace(go.Scatter(x=target[-days:].pd_dataframe().index, y=target[-days:].pd_dataframe().Close, name=f'Past {days} days'))
fig.add_trace(go.Scatter(x=predx.pd_dataframe().index, y=predx.pd_dataframe().Close, name=f'{forecast} day forecast'))
fig.update_layout(hovermode='x', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
st.plotly_chart(fig)

# Display forecast table
st.write(f"Table of {stock}'s {forecast}-day Forecasted Values")
table = predx.pd_dataframe()
st.write(table)
st.write('Note this app is for educational purposes, you can compare how these stocks fair to the forecasted values')
st.write('This app is NOT to be used to make investment decisions')
