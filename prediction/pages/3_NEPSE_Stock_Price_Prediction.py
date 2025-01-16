import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

# Load the dataset
predictions_df = pd.read_csv(r"F:\projectIII\nepse-predictions.csv")

# Function to display raw data for the selected ticker
def display_raw_data(ticker_name, predictions_df):
    ticker_data = predictions_df[predictions_df['Ticker'] == ticker_name]
    if ticker_data.empty:
        st.write(f"No data found for ticker {ticker_name}")
    else:
        st.write(ticker_data.head())

# Function to plot Original vs Predicted Price
def plot_original_vs_predicted(ticker_name, predictions_df):
    ticker_data = predictions_df[predictions_df['Ticker'] == ticker_name]
    if ticker_data.empty:
        st.write(f"No data found for ticker {ticker_name}")
        return
    
    ticker_data['date'] = pd.to_datetime(ticker_data['date'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(ticker_data['date'], ticker_data['Original Price'], label="Original Price", color='blue', marker='o', markersize=4)
    plt.plot(ticker_data['date'], ticker_data['Predicted Price'], label="Predicted Price", color='orange', marker='o', markersize=4)
    
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.title(f"{ticker_name}: Original Price vs Predicted Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Function to plot Original Price vs MA_3
def plot_original_vs_ma3(ticker_name, predictions_df):
    ticker_data = predictions_df[predictions_df['Ticker'] == ticker_name]
    if ticker_data.empty:
        st.write(f"No data found for ticker {ticker_name}")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(ticker_data['date'], ticker_data['Original Price'], label="Original Price", color='blue')
    plt.plot(ticker_data['date'], ticker_data['MA_3'], label="MA_3", color='green')
    plt.title(f"{ticker_name}: Original Price vs MA_3")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Function to plot Original Price vs MA_5
def plot_original_vs_ma5(ticker_name, predictions_df):
    ticker_data = predictions_df[predictions_df['Ticker'] == ticker_name]
    if ticker_data.empty:
        st.write(f"No data found for ticker {ticker_name}")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(ticker_data['date'], ticker_data['Original Price'], label="Original Price", color='blue')
    plt.plot(ticker_data['date'], ticker_data['MA_5'], label="MA_5", color='green')
    plt.title(f"{ticker_name}: Original Price vs MA_5")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Function to plot Original Price vs MA_7
def plot_original_vs_ma7(ticker_name, predictions_df):
    ticker_data = predictions_df[predictions_df['Ticker'] == ticker_name]
    if ticker_data.empty:
        st.write(f"No data found for ticker {ticker_name}")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(ticker_data['date'], ticker_data['Original Price'], label="Original Price", color='blue')
    plt.plot(ticker_data['date'], ticker_data['MA_7'], label="MA_7", color='red')
    plt.title(f"{ticker_name}: Original Price vs MA_7")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Streamlit UI
st.title("NEPSE Stock Price Predictions")

# Get unique tickers
tickers = predictions_df['Ticker'].unique()

# Dropdown for selecting ticker
ticker_name = st.selectbox("Select Ticker", tickers)

# Display raw data
st.subheader("Raw Data")
display_raw_data(ticker_name, predictions_df)

# Plot graphs
st.subheader(f"{ticker_name} - Original Price vs Predicted Price")
plot_original_vs_predicted(ticker_name, predictions_df)

# st.subheader(f"{ticker_name} - Original Price vs MA_3")
# plot_original_vs_ma3(ticker_name, predictions_df)

# st.subheader(f"{ticker_name} - Original Price vs MA_5")
# plot_original_vs_ma5(ticker_name, predictions_df)

# st.subheader(f"{ticker_name} - Original Price vs MA_7")
# plot_original_vs_ma7(ticker_name, predictions_df)
