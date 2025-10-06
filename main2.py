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
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365


# ---------- بارگذاری داده ----------
@st.cache_data
def load_data(ticker):
    # auto_adjust=False برای حفظ ساختار استاندارد داده‌ها
    data = yf.download(ticker, START, TODAY, auto_adjust=False)
    data.reset_index(inplace=True)
    # اطمینان از اینکه ستون تاریخ از نوع datetime است
    data['Date'] = pd.to_datetime(data['Date']).dt.floor('d')
    return data


data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("✅ Data loaded successfully!")

st.subheader("Raw data (last 5 rows)")
st.write(data.tail())

# بررسی نوع داده‌ها برای اطمینان
st.write("📊 Data types:")
st.write(data.dtypes)


# ---------- رسم نمودار ----------
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.update_layout(
        title_text="Time Series Data",
        xaxis_rangeslider_visible=True,
        template="plotly_white"
    )
    st.plotly_chart(fig)


if not data.empty:
    plot_raw_data()
else:
    st.error("⚠️ No data loaded! Please check your internet or ticker symbol.")


# ---------- آماده‌سازی داده برای Prophet ----------
df_train = data[['Date', 'Close']].copy()

# اطمینان از اینکه مقدار Close عددی است
df_train['Close'] = pd.to_numeric(df_train['Close'], errors='coerce')

# حذف مقادیر خالی
df_train = df_train.dropna(subset=['Close'])

# تغییر نام برای Prophet
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# اطمینان از نوع داده‌ها
df_train['ds'] = pd.to_datetime(df_train['ds'])
df_train['y'] = df_train['y'].astype(float)

# نمایش چک اولیه
st.write("✅ Prophet input sample:")
st.write(df_train.head())
st.write(df_train.dtypes)


# ---------- پیش‌بینی ----------
m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# ---------- نمایش داده‌های پیش‌بینی ----------
st.subheader("Forecast data (last 5 rows)")
st.write(forecast.tail())

# ---------- نمودار پیش‌بینی ----------
st.subheader("Forecast chart")
fig2 = plot_plotly(m, forecast)
st.plotly_chart(fig2)

# ---------- اجزای پیش‌بینی (ترند و فصلی) ----------
st.subheader("Forecast components")
fig3 = m.plot_components(forecast)
st.write(fig3)
