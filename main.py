import streamlit as st
from datetime import date
import pandas as pd

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go



# ---------- تنظیمات اولیه ----------
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("📈 Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select dateset for prediction", stocks)

n_years = st.slider("Years of prediction: ", 1, 4)
period = n_years * 365

# ---------- بارگذاری داده ----------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    # محور زمان روزانه بدون میلی‌ثانیه و ساعت
    # data['Date'] = pd.to_datetime(data['Date']).dt.floor('d')
    # data['Date'] = data['Date'].dt.date
    return data


data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("✅ Data loaded successfully!")

st.subheader("Raw data (last 5 rows)")
st.write(data.tail())


# ---------- رسم نمودار ----------
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'], name='stock open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='stock close'))
    fig.update_layout(title_text="Time Series Data", xaxis_rangeslider_visible =True, template="plotly_white")
    # fig.layout.update
    st.plotly_chart(fig)
  
    


if not data.empty:
    plot_raw_data()
else:
    st.error("⚠️ No data loaded! Please check your internet or ticker symbol.")


# ---------- پیش‌بینی ----------
# Forecasting
df_train = data[['Date','Close']].copy()
print(type(df_train))
df_train= df_train.rename(columns={"Date":"ds","Close":"y"})

# اطمینان از نوع صحیح
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train['ds'] = pd.to_datetime(df_train['ds'])
df_train = df_train.dropna()  # حذف مقادیر خالی

# ساخت مدل Prophet
m =Prophet()
m.fit(df_train)

# پیش‌بینی
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# ---------- نمایش داده‌های پیش‌بینی ----------
st.subheader("Forecast data (last 5 rows)")
st.write(forecast.tail())

# نمودار پیش‌بینی
st.subheader("Forecast chart")
fig2 = plot_plotly(m, forecast)
st.plotly_chart(fig2)

st.write('forecast components')
fig3 = m.plot_components(forecast)
st.write(fig3)