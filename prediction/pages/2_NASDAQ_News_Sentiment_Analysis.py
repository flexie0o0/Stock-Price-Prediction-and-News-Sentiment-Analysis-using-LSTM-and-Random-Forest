import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load LSTM model
@st.cache_resource
def load_lstm_model():
    return load_model(r"F:\projectIII\jup-notebook\Stock Predictions Model.keras")

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    stock_data['datetime'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data[['datetime', 'Close']].dropna()
    return stock_data

# Preprocess data for LSTM
def prepare_lstm_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(stock_data[['Close']])
    lstm_x = [scaled_prices[i-100:i] for i in range(100, len(scaled_prices))]
    return np.array(lstm_x), scaler

# Generate predictions
def process_predictions(stock_data, lstm_model):
    lstm_x, scaler = prepare_lstm_data(stock_data)
    lstm_predictions = lstm_model.predict(lstm_x)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    lstm_predicted_data = pd.DataFrame({
        'datetime': stock_data['datetime'][100:].values,
        'lstm_predicted_price': lstm_predictions.flatten()
    })
    sentiment_data = pd.read_csv(r"F:\projectIII\jup-notebook\random-forest-sentiment-NASDAQ.csv")
    sentiment_data['datetime'] = pd.to_datetime(sentiment_data['datetime'])
    merged_data = pd.merge_asof(
        lstm_predicted_data.sort_values('datetime'),
        sentiment_data.sort_values('datetime'),
        on='datetime',
        direction='nearest'
    )
    return merged_data

# Function to train Random Forest and add predictions
def add_rf_predictions(merged_data):
    sentiment_features = ['noisy_sentiment', 'sentiment_squared', 'sentiment_scaled', 'sentiment_boundary_dist']
    rf_data = merged_data.dropna(subset=sentiment_features)
    X = rf_data[sentiment_features]
    y = rf_data['lstm_predicted_price']
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_predictions = rf_model.predict(X)
    rf_data['rf_predicted_price'] = rf_predictions
    rf_data['combined_price'] = 0.7 * rf_data['lstm_predicted_price'] + 0.3 * rf_data['rf_predicted_price']
    return rf_data, rf_model

# Plot functions
def plot_lstm_rf(data):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data['datetime'], data['lstm_predicted_price'], label='LSTM Predictions', color='red')
    ax.plot(data['datetime'], data['rf_predicted_price'], label='RF Predictions', color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)


def plot_predictions(data):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data['datetime'], data['lstm_predicted_price'], label='LSTM Predictions', color='red')
    ax.plot(data['datetime'], data['rf_predicted_price'], label='RF Predictions', color='blue')
    ax.plot(data['datetime'], data['combined_price'], label='Combined Predictions', color='purple')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

def plot_feature_importance(rf_model, features):
    importances = rf_model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(features, importances, color='skyblue')
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

def plot_residuals(data):
    data['residuals'] = data['rf_predicted_price'] - data['lstm_predicted_price']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['datetime'], data['residuals'], alpha=0.6, color='red')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title("Residuals (Error) Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

def plot_sentiment_trends(data):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data['datetime'], data['noisy_sentiment'], label='Sentiment Score', color='orange', alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Score")
    ax.legend()
    st.pyplot(fig)

# Function to display news for the given ticker
def display_news_for_ticker(ticker):
    news_data = pd.read_csv(r"F:\pythonenaa\web-scrape-nasdaq-news\nasdaq_news_sentiment.csv")
    filtered_news = news_data[news_data['ticker'].str.upper() == ticker.upper()]
    
    st.write("### News Related to the Selected Ticker")
    if not filtered_news.empty:
        for _, row in filtered_news.iterrows():
            st.subheader(row['title'])
            st.write(f"**Date:** {row['datetime']}")
            st.write(f"**Sentiment:** {row['top_sentiment'].capitalize()}")
            st.write("---")
    else:
        st.write("No news articles found for the selected ticker.")

# Main app
def main():
    st.title("Stock Price Prediction with Sentiment Analysis")
    news_data = pd.read_csv(r"F:\pythonenaa\web-scrape-nasdaq-news\nasdaq_news_sentiment.csv")
    unique_tickers = sorted(news_data['ticker'].unique())
    
    # Inputs
    ticker = st.selectbox("Select stock ticker:", unique_tickers)
    start_date = st.date_input("Start date:", value=pd.to_datetime("2024-01-01"))
    end_date = st.date_input("End date:", value=pd.to_datetime("2024-10-30"))
    
    if st.button("Run Analysis"):
        try:
            st.write("Fetching stock data...")
            stock_data = fetch_stock_data(ticker, start_date, end_date)
            # st.write("Stock data preview:")
            # st.dataframe(stock_data.head())
            
            st.write("Loading LSTM model...")
            lstm_model = load_lstm_model()
            
            st.write("Processing predictions...")
            merged_data = process_predictions(stock_data, lstm_model)
            
            st.write("Training Random Forest...")
            processed_data, rf_model = add_rf_predictions(merged_data)
            
            # st.write("### Prediction and Analysis Results")
            # st.dataframe(processed_data.head())
            
            # Plot charts

            st.write('### LSTM and RF Predictions')
            plot_lstm_rf(processed_data)

            st.write("### Total Predictions")
            plot_predictions(processed_data)
            
            st.write("### Feature Importance")
            plot_feature_importance(rf_model, ['noisy_sentiment', 'sentiment_squared', 'sentiment_scaled', 'sentiment_boundary_dist'])
            
            st.write("### Residuals Over Time")
            plot_residuals(processed_data)
            
            st.write("### Sentiment Trends")
            plot_sentiment_trends(processed_data)

            st.write("")
            st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  # Horizontal line as a divider
            st.write("")
            
            # Display news for the ticker
            display_news_for_ticker(ticker)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
