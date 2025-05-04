import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[["Close"]].dropna()
    return df

def preprocess_data(df, seq_len):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main(ticker, start, end, seq_len):
    print(f"🔄 Downloading data for {ticker} from {start} to {end}...")
    df = download_data(ticker, start, end)

    print("📦 Preprocessing data...")
    X, y, scaler = preprocess_data(df, seq_len)

    print("🧠 Building model...")
    model = build_model((X.shape[1], X.shape[2]))

    print("🏋️ Training model...")
    early_stop = EarlyStopping(monitor="loss", patience=5)
    model.fit(X, y, epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)

    # Create models folder
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{ticker.upper()}_model.h5")

    print(f"💾 Saving model to {model_path}")
    model.save(model_path)

    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock price prediction model.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--seq_len", type=int, default=60, help="Sequence length (default: 60)")

    args = parser.parse_args()
    main(args.ticker, args.start, args.end, args.seq_len)
