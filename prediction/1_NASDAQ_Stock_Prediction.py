# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained model
model = load_model(r'F:\projectIII\jup-notebook\Stock Predictions Model.keras')

# Custom CSS for styling
st.markdown("""
    <style>
        .chart-container {
            border: 2px solid gray;
            border-radius: 15px;
            padding: 3px;
            margin-bottom: 20px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .center-title {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App Title
st.markdown("<h1 class='center-title'>NASDAQ Stock Market Predictor</h1>", unsafe_allow_html=True)
# st.markdown("""
#     <script>
#         document.addEventListener('DOMContentLoaded', function() {
#             var input = document.querySelector("input[type='text']");
#             input.addEventListener('input', function() {
#                 this.value = this.value.toUpperCase();
#             });
#         });
#     </script>
# """, unsafe_allow_html=True)

# User Input for Stock Symbol
stock = st.text_input("Enter Stock Symbol", "GOOG")
start_date = '2014-01-01'
end_date = '2024-10-30'

# Fetch Stock Data
data = yf.download(stock, start=start_date, end=end_date)

# Display Stock Data
st.subheader("Stock Data")
st.write(data)

# Prepare Data for Training and Testing
data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):len(data)])

# data Scaling
scaler = MinMaxScaler(feature_range=(0, 1))

# Append last 100 days of training data to test data for continuity
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

# Moving Averages Visualization
st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.markdown("<h3 class='center-title'>Price vs 50-Day Moving Average</h3>", unsafe_allow_html=True)
ma_50 = data['Close'].rolling(window=50).mean()
fig1 = plt.figure(figsize=(10, 6))
plt.plot(ma_50, 'r', label="MA 50 Days")
plt.plot(data['Close'], 'g', label="Close Price")
plt.legend()
st.pyplot(fig1)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.markdown("<h3 class='center-title'>Price vs 50-Day and 100-Day Moving Averages</h3>", unsafe_allow_html=True)
ma_100 = data['Close'].rolling(window=100).mean()
fig2 = plt.figure(figsize=(10, 6))
plt.plot(ma_50, 'r', label="MA 50 Days")
plt.plot(ma_100, 'b', label="MA 100 Days")
plt.plot(data['Close'], 'g', label="Close Price")
plt.legend()
st.pyplot(fig2)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.markdown("<h3 class='center-title'>Price vs 100-Day and 200-Day Moving Averages</h3>", unsafe_allow_html=True)
ma_200 = data['Close'].rolling(window=200).mean()
fig3 = plt.figure(figsize=(10, 6))
plt.plot(ma_100, 'r', label="MA 100 Days")
plt.plot(ma_200, 'b', label="MA 200 Days")
plt.plot(data['Close'], 'g', label="Close Price")
plt.legend()
st.pyplot(fig3)
st.markdown("</div>", unsafe_allow_html=True)

# Prepare Test Data for Prediction
X_test = []
y_test = []

for i in range(100, data_test_scaled.shape[0]):
    X_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Make Predictions
predictions = model.predict(X_test)

# Inverse Scale Predictions and Actual Values
scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y_test = y_test * scale_factor

# Plot Predictions vs Actual Prices
st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.markdown("<h3 class='center-title'>Original Price vs Predicted Price</h3>", unsafe_allow_html=True)
fig4 = plt.figure(figsize=(10, 6))
plt.plot(predictions, 'r', label="Predicted Price")
plt.plot(y_test, 'g', label="Original Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig4)
st.markdown("</div>", unsafe_allow_html=True)
