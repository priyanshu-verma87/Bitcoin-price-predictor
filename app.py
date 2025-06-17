import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("Bitcoin Price Predictor App")

stock = "BTC-USD"

# Download historical data
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)
bit_coin_data = yf.download(stock, start, end)

bit_coin_data.columns.name = None

# Load the pre-trained model
model = load_model("Latest_bitcoin_model.keras")

# Display data
st.subheader("Bitcoin Data")
st.write(bit_coin_data)

# Split into train/test
splitting_len = int(len(bit_coin_data) * 0.9)
x_test = pd.DataFrame(bit_coin_data.Close[splitting_len:])
x_test.columns.name = None

# Plot full data
st.subheader('Original Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(bit_coin_data.Close, 'b')
st.pyplot(fig)

# Plot test data
st.subheader("Test Close Price")
st.write(x_test)

fig = plt.figure(figsize=(15, 6))
plt.plot(x_test, 'b')
st.pyplot(fig)

# Scale test data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test.values)

# Create 100-step input sequences
x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Predict
predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Prepare DataFrame for visualization
ploting_data = pd.DataFrame({
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
}, index=bit_coin_data.index[splitting_len + 100:])

# Display predictions
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Plot predictions vs real data
st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([bit_coin_data.Close[:splitting_len + 100], ploting_data], axis=0))
plt.legend(["Data - not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

# Future predictions
st.subheader("Future Price values")
last_100 = bit_coin_data[['Close']].tail(100)
last_100_scaled = scaler.fit_transform(last_100.values).reshape(1, -1, 1)

def predict_future(no_of_days, prev_100_scaled):
    future_predictions = []
    prev_100_flat = prev_100_scaled.reshape(-1).tolist()

    for _ in range(no_of_days):
        input_seq = np.array(prev_100_flat[-100:]).reshape(1, 100, 1)
        next_day_scaled = model.predict(input_seq)[0][0]
        prev_100_flat.append(next_day_scaled)

        # Convert back to actual price
        next_day_unscaled = scaler.inverse_transform([[next_day_scaled]])[0][0]
        future_predictions.append(next_day_unscaled)

    return future_predictions

no_of_days = int(st.text_input("Enter the number of future days to predict:", "10"))
future_results = predict_future(no_of_days, last_100_scaled)
future_results = np.array(future_results).reshape(-1, 1)

# Plot future predictions
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.DataFrame(future_results), marker='o')
for i in range(len(future_results)):
    plt.text(i, future_results[i], int(future_results[i][0]))
plt.xlabel('Days from Today')
plt.ylabel('Close Price')
plt.xticks(range(no_of_days))
plt.yticks(range(int(min(future_results)), int(max(future_results)) + 100, 100))
plt.title('Predicted Future Closing Prices')
st.pyplot(fig)
