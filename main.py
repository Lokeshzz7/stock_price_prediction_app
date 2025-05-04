import streamlit as st

from datetime import date

import yfinance as yf

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

start = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")


st.title("Stock Prediction")

stocks = ('AAPL','GOOG','MSFT','GME')

select = st.selectbox("Select Data",stocks)

n_years = st.slider("Years of Prediction",1,4)

period = n_years * 365

