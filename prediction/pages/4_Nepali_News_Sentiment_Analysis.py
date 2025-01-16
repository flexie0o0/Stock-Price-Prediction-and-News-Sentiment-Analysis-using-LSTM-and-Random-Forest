import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

# Load the datasets
predictions_df = pd.read_csv(r"F:\projectIII\nepse-predictions.csv")
sentiment_df = pd.read_csv(r"F:\projectIII\jup-notebook\random-forest-sentiment-NEPSE.csv")

# Function to display raw data for the selected ticker
def display_raw_data(ticker_name, predictions_df, sentiment_df):
    ticker_predictions = predictions_df[predictions_df['Ticker'] == ticker_name]
    ticker_sentiments = sentiment_df[sentiment_df['ticker'] == ticker_name]

    if ticker_predictions.empty and ticker_sentiments.empty:
        st.write(f"No data found for ticker {ticker_name}")
    else:
        st.subheader("Predictions Data")
        st.write(ticker_predictions.head())
        
        st.subheader("Sentiment Data")
        st.write(ticker_sentiments.head())

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

# Function to plot Original Price vs Moving Averages
def plot_original_vs_ma(ticker_name, predictions_df, ma_column, ma_label):
    ticker_data = predictions_df[predictions_df['Ticker'] == ticker_name]
    if ticker_data.empty:
        st.write(f"No data found for ticker {ticker_name}")
        return
    
    ticker_data['date'] = pd.to_datetime(ticker_data['date'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(ticker_data['date'], ticker_data['OriginalPrice'], label="Original Price", color='blue')
    plt.plot(ticker_data['date'], ticker_data[ma_column], label=ma_label, color='green')
    plt.title(f"{ticker_name}: Original Price vs {ma_label}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Function to plot Sentiment Impact
def plot_sentiment_impact(ticker_name, sentiment_df):
    ticker_data = sentiment_df[sentiment_df['ticker'] == ticker_name]
    if ticker_data.empty:
        st.write(f"No sentiment data found for ticker {ticker_name}")
        return
    
    ticker_data['date'] = pd.to_datetime(ticker_data['date'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(ticker_data['date'], ticker_data['sentiment_score'], label="Sentiment Score", color='purple', marker='o', markersize=4)
    plt.title(f"{ticker_name}: Sentiment Score Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

def display_news_for_ticker(ticker_name):
    news_data = sentiment_df
    filtered_news = news_data[news_data['ticker'].str.upper() == ticker_name.upper()]
    
    # st.write("### News Related to the Selected Ticker")
    if not filtered_news.empty:
        for _, row in filtered_news.iterrows():
            st.subheader(row['title'])
            st.write(f"**Date:** {row['date']}")
            st.write(f"**Sentiment:** {row['top_sentiment'].capitalize()}")
            st.write("---")
    else:
        st.write("No news articles found for the selected ticker.")

# Streamlit UI
st.title("NEPSE Stock Price Predictions and Sentiment Analysis")

# Get unique tickers
tickers = predictions_df['Ticker'].unique()

# Dropdown for selecting ticker
ticker_name = st.selectbox("Select Ticker", tickers)

# Display raw data
st.subheader("Raw Data")
display_raw_data(ticker_name, predictions_df, sentiment_df)

# Plot graphs
st.subheader(f"{ticker_name} - Original Price vs Predicted Price")
plot_original_vs_predicted(ticker_name, predictions_df)

# st.subheader(f"{ticker_name} - Original Price vs MA_3")
# plot_original_vs_ma(ticker_name, predictions_df, 'MA_3', 'MA_3')

# st.subheader(f"{ticker_name} - Original Price vs MA_5")
# plot_original_vs_ma(ticker_name, predictions_df, 'MA_5', 'MA_5')

st.subheader(f"{ticker_name} - Sentiment Score Over Time")
plot_sentiment_impact(ticker_name, sentiment_df)

st.write("")
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  # Horizontal line as a divider
st.write("")

st.subheader(f"{ticker_name} - News and its Sentiments")
display_news_for_ticker(ticker_name)
