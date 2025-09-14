# Real-Time-Stock-Market-Dashboard
Dashboard that tracks and visualizes live stock market data.
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import time

# Set the title of the Streamlit application
st.title("Real-Time Stock Market Dashboard")

# --- User Input Widgets ---
ticker_symbol = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "AAPL").upper()

timeframe_options = {
    "1 Day": "TIME_SERIES_INTRADAY", # Requires interval parameter
    "1 Week": "TIME_SERIES_DAILY", # Use Daily and filter
    "1 Month": "TIME_SERIES_DAILY", # Use Daily and filter
    "1 Year": "TIME_SERIES_DAILY", # Use Daily and filter
    "Full History": "TIME_SERIES_DAILY" # Use Daily
}
selected_timeframe_label = st.selectbox("Select Timeframe", list(timeframe_options.keys()))
selected_timeframe_function = timeframe_options[selected_timeframe_label]

# For intraday, allow selecting interval
interval = None
if selected_timeframe_function == "TIME_SERIES_INTRADAY":
    interval = st.selectbox("Select Interval (Intraday)", ["1min", "5min", "15min", "30min", "60min"])

indicator_options = {
    "Simple Moving Average (SMA)": "SMA",
    "Exponential Moving Average (EMA)": "EMA",
    "Relative Strength Index (RSI)": "RSI"
    # Add more indicators as needed
}
selected_indicators = st.multiselect("Select Indicators to Display", list(indicator_options.keys()))

# --- Placeholders for Chart and Indicators ---
chart_placeholder = st.empty()
indicators_placeholder = st.empty()

# --- Data Fetching Function ---
def get_stock_data(symbol, function, interval=None, outputsize="compact"):
    # Replace 'YOUR_ALPHA_VANTAGE_API_KEY' with your actual API key
    # It's recommended to store API keys securely, e.g., in Streamlit Secrets
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', 'YOUR_ALPHA_VANTAGE_API_KEY')
    base_url = "https://www.alphavantage.co/query?"

    params = {
        "function": function,
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": outputsize # 'compact' returns the last 100 points, 'full' returns full history
    }
    if interval:
        params["interval"] = interval

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None

# --- Data Processing Function ---
def process_stock_data(data, function):
    df = pd.DataFrame()
    time_series_key = None

    if function == "TIME_SERIES_INTRADAY":
        # Find the correct time series key based on interval
        time_series_key = next((key for key in data.keys() if 'Time Series (' in key), None)
    elif function == "TIME_SERIES_DAILY" and "Time Series (Daily)" in data:
        time_series_key = "Time Series (Daily)"

    if time_series_key and time_series_key in data:
        raw_data = data[time_series_key]
        df = pd.DataFrame.from_dict(raw_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })
        df = df.astype(float)
        df = df.sort_index() # Sort by date

    elif data and "Information" in data:
        st.warning(data["Information"])
        st.warning("Please replace 'YOUR_ALPHA_VANTAGE_API_KEY' with a valid Alpha Vantage API key.")

    elif data and "Error Message" in data:
         st.error(f"API Error: {data['Error Message']}")

    else:
        st.warning("No stock data found for the selected parameters.")
        if data: # Only print data if it's not None or empty
            st.json(data) # Display raw data for debugging


    return df

# --- Indicator Calculation Functions ---
def calculate_sma(df, window=20):
    df['SMA'] = df['close'].rolling(window=window).mean()
    return df

def calculate_ema(df, window=20):
    df['EMA'] = df['close'].ewm(span=window, adjust=False).mean()
    return df

def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Handle division by zero in RS calculation
    rs = gain.ewm(span=window, adjust=False).mean() / loss.ewm(span=window, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# --- Main Dashboard Update Logic ---
# Use Streamlit's session state to store historical data across reruns
# This is a simplified approach; for true real-time updates, you might need
# a background process or a different architecture.
if 'stock_history_df' not in st.session_state or \
   st.session_state.get('ticker_symbol') != ticker_symbol or \
   st.session_state.get('selected_timeframe_label') != selected_timeframe_label or \
   st.session_state.get('interval') != interval:

    # Fetch new data if ticker, timeframe, or interval changes
    st.session_state.ticker_symbol = ticker_symbol
    st.session_state.selected_timeframe_label = selected_timeframe_label
    st.session_state.interval = interval

    output_size = "compact" # Default to compact
    if selected_timeframe_label == "Full History":
        output_size = "full" # Request full history for this option

    stock_data = get_stock_data(ticker_symbol, selected_timeframe_function, interval=interval, outputsize=output_size)
    st.session_state.stock_history_df = process_stock_data(stock_data, selected_timeframe_function)

# Apply timeframe filter if not "Full History"
filtered_df = st.session_state.stock_history_df.copy()
if not filtered_df.empty and selected_timeframe_label != "Full History":
    end_date = filtered_df.index.max()
    if selected_timeframe_label == "1 Day":
        start_date = end_date - pd.Timedelta(days=1)
    elif selected_timeframe_label == "1 Week":
        start_date = end_date - pd.Timedelta(weeks=1)
    elif selected_timeframe_label == "1 Month":
        start_date = end_date - pd.Timedelta(days=30) # Approximation
    elif selected_timeframe_label == "1 Year":
        start_date = end_date - pd.Timedelta(days=365) # Approximation
    else:
        start_date = filtered_df.index.min() # Should not happen with current options

    filtered_df = filtered_df[filtered_df.index >= start_date]

# Calculate selected indicators on the filtered data
if not filtered_df.empty:
    for indicator in selected_indicators:
        if indicator == "Simple Moving Average (SMA)":
            filtered_df = calculate_sma(filtered_df)
        elif indicator == "Exponential Moving Average (EMA)":
            filtered_df = calculate_ema(filtered_df)
        elif indicator == "Relative Strength Index (RSI)":
             filtered_df = calculate_rsi(filtered_df)
        # Add more indicator calculations here

    # Update Chart
    if not filtered_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['close'], mode='lines', name='Close Price'))

        # Add selected indicator traces
        for indicator in selected_indicators:
             if indicator == "Simple Moving Average (SMA)" and 'SMA' in filtered_df.columns:
                 fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['SMA'], mode='lines', name='SMA'))
             elif indicator == "Exponential Moving Average (EMA)" and 'EMA' in filtered_df.columns:
                 fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['EMA'], mode='lines', name='EMA'))
             # RSI is typically plotted on a separate subplot, handle below

        fig.update_layout(
            title=f"{ticker_symbol} Stock Price and Indicators ({selected_timeframe_label})",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis=dict(tickformat='%Y-%m-%d') # Adjust format based on timeframe
        )

        # Display RSI separately if selected
        if "Relative Strength Index (RSI)" in selected_indicators and 'RSI' in filtered_df.columns:
             fig_rsi = go.Figure()
             fig_rsi.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['RSI'], mode='lines', name='RSI'))
             fig_rsi.update_layout(title="Relative Strength Index (RSI)", yaxis_title="RSI Value")
             chart_placeholder.plotly_chart(fig, use_container_width=True)
             st.plotly_chart(fig_rsi, use_container_width=True) # Display RSI below the main chart
        else:
             chart_placeholder.plotly_chart(fig, use_container_width=True)


    # Update Indicators Display (show latest values)
    indicators_text = "Latest Indicators:<br>"
    if not filtered_df.empty:
        latest_data = filtered_df.iloc[-1]
        for indicator in selected_indicators:
            indicator_col = None
            if indicator == "Simple Moving Average (SMA)" and 'SMA' in latest_data:
                 indicator_col = 'SMA'
            elif indicator == "Exponential Moving Average (EMA)" and 'EMA' in latest_data:
                 indicator_col = 'EMA'
            elif indicator == "Relative Strength Index (RSI)" and 'RSI' in latest_data:
                 indicator_col = 'RSI'

            if indicator_col and pd.notna(latest_data.get(indicator_col)): # Use .get() for safer access
                 indicators_text += f"- {indicator}: {latest_data[indicator_col]:.2f}<br>"
            elif indicator_col:
                 indicators_text += f"- {indicator}: N/A (Insufficient data)<br>"

        if indicators_text == "Latest Indicators:<br>": # If no indicators selected or available
             indicators_placeholder.write("No indicators selected or available.")
        else:
             indicators_placeholder.markdown(indicators_text) # Use markdown to render HTML

    else:
        indicators_placeholder.write("No data available to calculate or display indicators.")
