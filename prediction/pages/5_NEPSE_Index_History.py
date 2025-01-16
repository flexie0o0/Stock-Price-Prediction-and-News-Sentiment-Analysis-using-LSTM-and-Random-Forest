# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the processed stock data from CSV
df = pd.read_csv("F:\projectIII\jup-notebook\export.csv")

# Title of the App
st.title("NEPSE Stock Analysis of Index History Data")

# Display raw data
st.subheader("Stock Data (Raw)")
st.write(df)

# Moving Averages Calculation
df['MA50'] = df['close'].rolling(window=50).mean()
df['MA100'] = df['close'].rolling(window=100).mean()
df['MA200'] = df['close'].rolling(window=200).mean()

# Fill missing values in the moving averages by interpolation
df['MA50'] = df['MA50'].interpolate()
df['MA100'] = df['MA100'].interpolate()
df['MA200'] = df['MA200'].interpolate()


# Custom CSS styling for chart containers
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

# Chart 1: Original Price vs MA50
st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.markdown("<h3 class='center-title'>Original Price vs 50-Day Moving Average</h3>", unsafe_allow_html=True)
fig1 = plt.figure(figsize=(10, 6))
plt.plot(df['close'], color='blue', label="Close Price")
plt.plot(df['MA50'], color='red', label="MA 50")
plt.xlabel("Time (Days)")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig1)
st.markdown("</div>", unsafe_allow_html=True)

# Chart 2: MA50 vs MA100
st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.markdown("<h3 class='center-title'>50-Day vs 100-Day Moving Averages</h3>", unsafe_allow_html=True)
fig2 = plt.figure(figsize=(10, 6))
plt.plot(df['MA50'], color='red', label="MA 50")
plt.plot(df['MA100'], color='green', label="MA 100")
plt.xlabel("Time (Days)")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
st.markdown("</div>", unsafe_allow_html=True)

# Chart 3: MA100 vs MA200
st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.markdown("<h3 class='center-title'>100-Day vs 200-Day Moving Averages</h3>", unsafe_allow_html=True)
fig3 = plt.figure(figsize=(10, 6))
plt.plot(df['MA100'], color='green', label="MA 100")
plt.plot(df['MA200'], color='purple', label="MA 200")
plt.xlabel("Time (Days)")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig3)
st.markdown("</div>", unsafe_allow_html=True)

# Volume plot
st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.markdown("<h3 class='center-title'>Stock Volume Over Time</h3>", unsafe_allow_html=True)
fig4 = plt.figure(figsize=(10, 6))
plt.plot(df['tradedShares'], color='black', label="Volume")
plt.xlabel("Time (Days)")
plt.ylabel("Volume")
plt.legend()
st.pyplot(fig4)
st.markdown("</div>", unsafe_allow_html=True)

# Additional custom charts if any
# For example: Open vs Close Price
# st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
# st.markdown("<h3 class='center-title'>Open vs Close Prices</h3>", unsafe_allow_html=True)
# fig5 = plt.figure(figsize=(10, 6))
# plt.plot(df['open'], color='red', label="Open Price", linewidth=1.5, marker='o', markersize=2)
# plt.plot(df['close'], color='green', label="Close Price", linewidth=1.5)

# plt.xlabel("Time (Days)")
# plt.ylabel("Price")
# plt.legend()
# st.pyplot(fig5)
# st.markdown("</div>", unsafe_allow_html=True)
