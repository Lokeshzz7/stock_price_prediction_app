import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

st.sidebar.title("🔧 Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
seq_len = st.sidebar.slider("Sequence Length", 10, 100, 60)

st.title("📈 Stock Price Prediction App")

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[["Close"]]
    df.dropna(inplace=True)
    return df

df = load_data(ticker, start_date, end_date)

if df.empty:
    st.error("No data found for the selected range.")
    st.stop()

with st.expander("🔍 View Raw Data"):
    st.write(df.tail())

st.subheader("📊 Stock Closing Price")
fig, ax = plt.subplots()
ax.plot(df.index, df["Close"])
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.set_title(f"{ticker} Stock Price")
st.pyplot(fig)

@st.cache_data
def preprocess_data(df, _scaler, seq_len):
    scaled = _scaler.fit_transform(df)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaled

scaler = MinMaxScaler()
X, y, scaled_data = preprocess_data(df, scaler, seq_len)

model_path = f"models/{ticker}_model.h5"
try:
    model = load_model(model_path)
except:
    st.warning("Trained model not found. Please train the model first using `train_model.py`.")
    st.stop()

predictions = model.predict(X)
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y)

st.subheader("📉 Predicted vs Actual Prices")
fig2, ax2 = plt.subplots()
ax2.plot(actual_prices, label="Actual Price")
ax2.plot(predicted_prices, label="Predicted Price")
ax2.legend()
st.pyplot(fig2)

st.success(f"📌 Latest Actual Price: ${actual_prices[-1].item():.2f}")
st.info(f"📌 Latest Predicted Price: ${predicted_prices[-1].item():.2f}")

pred_df = pd.DataFrame({
    "Actual": actual_prices.flatten(),
    "Predicted": predicted_prices.flatten()
})
csv = pred_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")

raw_csv = df.to_csv().encode("utf-8")
st.download_button("⬇️ Download Raw Stock Data", raw_csv, "stock_data.csv", "text/csv")

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
st.metric("Root Mean Squared Error", f"{rmse:.4f}")
