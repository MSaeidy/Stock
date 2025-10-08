import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


st.title("Stock Price Viewer — (5 years)")

stocks = (
    "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","TSLA","GOOG","BRK.B",
    "JPM","ORCL","WMT","LLY","V","MA","NFLX","XOM","JNJ","PLTR",
    "ABBV","COST","HD","AMD","BAC","PG","UNH","GE","CVX","KO",
    "CSCO","IBM","WFC","TMUS","MS","PM","GS","AMGN","ACN","TJX",
    "APH","SPGI","DHR","NEE","AMT","RTX","MCD","UBER","SHOP","CAT",
    "ICE","NDAQ"
)

ticker_symbol = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction: ", 1 , 4)
period = n_years * 365

#  --- loading ---
def load_data(ticker):
    if ticker:
        ticker_data = yf.Ticker(ticker)
        try:
            # فقط period — هیچ start یا end ای تعیین نشده
            data = ticker_data.history(period='5y')  # مثلا 5 سال اخیر
            # data['Date'] = pd.to_datetime(data['Date']).dt.floor('d')
            return data
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please enter a valid symbol")

data =load_data(ticker_symbol)

# --- chart ---
def c_chart(TDF):
    if not TDF.empty:
        st.subheader("Closing Price")
        st.line_chart(TDF['Close'])

        st.subheader("Volume")
        st.line_chart(TDF['Volume'])
    else:
        st.warning("No historical data available for this symbol / period.")


c_chart(data)


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index,y=data['Open'],name='Stock Open'))
    fig.add_trace(go.Scatter(x=data.index,y=data['Close'],name='Stock Close'))
    fig.update_layout(title_text = "Time Series Data" , xaxis_rangeslider_visible = True , )
    st.plotly_chart(fig)

plot_raw_data()    

st.subheader("Raw data ")
st.write(data.tail())

# --- ready to prophet ---
df_train = data.reset_index()[['Date','Close']]
df_train['Date'] = df_train['Date'].dt.tz_localize(None)
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast data")
st.write(forecast.tail())

st.subheader("Forecast Chart")
fig2 = plot_plotly(m,forecast)
st.plotly_chart(fig2)

st.subheader("Forecast Components")
fig3 = m.plot_components(forecast)
st.write(fig3)
