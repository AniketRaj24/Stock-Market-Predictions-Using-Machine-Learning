
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Stock Market Predictor", layout="centered")

# Load model from same directory
model = load_model("Stock_Predictions_Model.keras")


st.header("📈 Stock Market Predictor (LSTM Based)")

stock = st.text_input("Enter Stock Symbol", "GOOG")
start = "2012-01-01"
end = "2022-12-31"

with st.spinner("Fetching stock data..."):
    data = yf.download(stock, start, end)

if data.empty:
    st.error("Invalid stock symbol or no data available.")
    st.stop()

st.subheader("Latest Stock Data")
st.write(data.tail())

# -------- Train-Test Split --------
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test)

# -------- Moving Averages --------
st.subheader("Price vs MA50")
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, label="MA50")
plt.plot(data.Close, label="Close Price")
plt.legend()
st.pyplot(fig1)

st.subheader("Price vs MA50 vs MA100")
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, label="MA50")
plt.plot(ma_100_days, label="MA100")
plt.plot(data.Close, label="Close Price")
plt.legend()
st.pyplot(fig2)

st.subheader("Price vs MA100 vs MA200")
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, label="MA100")
plt.plot(ma_200_days, label="MA200")
plt.plot(data.Close, label="Close Price")
plt.legend()
st.pyplot(fig3)

# -------- Sequence Creation --------
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# -------- Prediction --------
predict = model.predict(x)

scale_factor = 1 / scaler.scale_[0]
predict = predict * scale_factor
y = y * scale_factor

rmse = np.sqrt(mean_squared_error(y, predict))

st.subheader("Original Price vs Predicted Price")
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, label="Original Price")
plt.plot(predict, label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig4)

st.success(f"Model RMSE: {rmse:.2f}")
