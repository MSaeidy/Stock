import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Preication App")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stocks = st.selectbox("Select dateset for prediction" , stocks)

n_years = st.slider("Years of prediction: ", 1 , 4)
period = n_years * 365