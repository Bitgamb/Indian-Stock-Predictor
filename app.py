import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Web Page Config ---
st.set_page_config(page_title="Indian Stock AI Predictor", layout="wide")
st.title("📈 Indian Stock AI Predictor & Dashboard")
st.markdown("Search any Indian stock to analyze technical signals, view fundamentals, and predict future trends using an LSTM Neural Network.")

# --- Function to fetch ALL NSE Tickers ---
@st.cache_data(ttl=86400) # Caches the data for 24 hours to speed up the app
def fetch_indian_tickers():
    # Fallback list just in case the NSE website blocks the cloud request
    fallback = {
        "Reliance Industries (RELIANCE)": "RELIANCE.NS",
        "Tata Consultancy Services (TCS)": "TCS.NS",
        "HDFC Bank (HDFCBANK)": "HDFCBANK.NS",
        "Infosys (INFY)": "INFY.NS",
        "State Bank of India (SBIN)": "SBIN.NS",
        "Bharti Airtel (BHARTIARTL)": "BHARTIARTL.NS",
        "Tata Motors (TATAMOTORS)": "TATAMOTORS.NS",
        "Zomato (ZOMATO)": "ZOMATO.NS"
    }
    try:
        # Official NSE Equity Master List
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df['Display'] = df['NAME OF COMPANY'] + " (" + df['SYMBOL'] + ")"
            df['Ticker'] = df['SYMBOL'] + ".NS"
            return dict(zip(df['Display'], df['Ticker']))
        else:
            return fallback
    except Exception:
        return fallback

# Load the dictionary of all stocks
nse_stocks = fetch_indian_tickers()

# --- User Input Section ---
st.sidebar.header("Stock Selection")

input_method = st.sidebar.radio(
    "Choose Input Method:",
    ("Search NSE Stocks (Auto-Dropdown)", "Enter Custom Ticker")
)

if input_method == "Search NSE Stocks (Auto-Dropdown)":
    selected_company = st.sidebar.selectbox("Search & Select a Company:", list(nse_stocks.keys()))
    ticker = nse_stocks[selected_company]
else:
    st.sidebar.info("Use this if you want a BSE stock. Add '.BO' at the end (e.g., PAYTM.BO)")
    ticker = st.sidebar.text_input("Enter Ticker Symbol:", value="ZOMATO.NS").upper()

period = st.sidebar.selectbox("Select Historical Data Period:", ["3mo", "6mo", "1y", "2y", "5y"], index=1)

backtest_days = st.sidebar.slider(
    "Backtest Period (Days)",
    min_value=30,
    max_value=180,
    value=60
)

# --- Analysis & Prediction ---
if st.sidebar.button("Analyze & Predict"):
    with st.spinner(f"Fetching data and company info for {ticker}..."):
        
        # 1. Fetch Company Info (Cloud-Safe)
        stock_info = {} # Prevents NameError if yfinance fails
        try:
            stock_info = yf.Ticker(ticker).info
            company_name = stock_info.get('longName', ticker)
            sector = stock_info.get('sector', '') # Changed to empty string if missing
        except:
            company_name = ticker
            sector = ""

        # 2. Data Fetching
        data = yf.download(ticker, period=period, interval="1d")
        
        if data.empty:
            st.error(f"No data found for '{ticker}'. Please check the ticker symbol.")
        else:
            data.dropna(inplace=True)
            
            # Data Processing
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data['Close'][ticker].squeeze()
            else:
                close_prices = data['Close'].squeeze()
            
            # Metrics Calculation
            current_price = close_prices.iloc[-1]
            previous_price = close_prices.iloc[-2] if len(close_prices) > 1 else current_price
            price_change = current_price - previous_price
            percentage_change = (price_change / previous_price) * 100
            
            # --- Robust 52-Week High/Low Calculation ---
            high_52w = stock_info.get('fiftyTwoWeekHigh')
            low_52w = stock_info.get('fiftyTwoWeekLow')
            
            if high_52w is None or low_52w is None:
                try:
                    history_1y = yf.Ticker(ticker).history(period="1y")
                    high_52w = history_1y['High'].max()
                    low_52w = history_1y['Low'].min()
                except:
                    high_52w = "N/A"
                    low_52w = "N/A"

            high_display = f"₹ {high_52w:.2f}" if isinstance(high_52w, (int, float)) else "N/A"
            low_display = f"₹ {low_52w:.2f}" if isinstance(low_52w, (int, float)) else "N/A"

            # --- UI: Top Dashboard Metrics ---
            # Cleanly handle missing sector info
            if sector:
                st.subheader(f"📊 {company_name} ({sector})")
            else:
                st.subheader(f"📊 {company_name}")
                
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Current Close Price", value=f"₹ {current_price:.2f}", delta=f"{price_change:.2f} ({percentage_change:.2f}%)")
            with col2:
                st.metric(label="52-Week High", value=high_display)
            with col3:
                st.metric(label="52-Week Low", value=low_display)
            
            st.divider()
            
            # --- Technical Indicators Calculation ---
            ma5 = close_prices.rolling(window=5).mean()
            ma10 = close_prices.rolling(window=10).mean()
            data['RSI'] = RSIIndicator(close_prices, window=14).rsi()
            data['MACD'] = MACD(close_prices).macd()
            data['MA5'] = ma5
            data['MA10'] = ma10
            data.dropna(inplace=True)

            data['Signal'] = np.where(data['MA5'] > data['MA10'], 1, 0)
            data['Position'] = data['Signal'].diff()

            # --- UI: Tabs for Technical Analysis ---
            st.subheader("⚙️ Technical Analysis")
            tab1, tab2, tab3 = st.tabs(["Moving Averages & Signals", "RSI & MACD", "Raw Data & Download"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(data.index, close_prices.loc[data.index], label='Close Price', color='gray', alpha=0.6)
                ax.plot(data.index, data['MA5'], label='MA5 (Short Trend)', color='blue', alpha=0.8)
                ax.plot(data.index, data['MA10'], label='MA10 (Long Trend)', color='red', alpha=0.8)
                
                buy_signals = data[data['Position'] == 1]
                sell_signals = data[data['Position'] == -1]
                ax.plot(buy_signals.index, data['MA5'][data['Position'] == 1], '^', markersize=12, color='green', label='Buy Signal')
                ax.plot(sell_signals.index, data['MA5'][data['Position'] == -1], 'v', markersize=12, color='red', label='Sell Signal')
                
                ax.set_ylabel("Price (INR)")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)

            with tab2:
                fig2, (ax_rsi, ax_macd) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                
                # RSI Plot
                ax_rsi.plot(data.index, data['RSI'], label='RSI (14)', color='purple')
                ax_rsi.axhline(70, linestyle='dashed', color='red', alpha=0.5)
                ax_rsi.axhline(30, linestyle='dashed', color='green', alpha=0.5)
                ax_rsi.set_ylabel("RSI")
                ax_rsi.legend(loc="upper left")
                ax_rsi.grid(alpha=0.3)

                # MACD Plot
                ax_macd.plot(data.index, data['MACD'], label='MACD', color='green')
                ax_macd.axhline(0, linestyle='dashed', color='gray', alpha=0.5)
                ax_macd.set_ylabel("MACD")
                ax_macd.legend(loc="upper left")
                ax_macd.grid(alpha=0.3)
                
                st.pyplot(fig2)

            with tab3:
                st.dataframe(data.tail(15)) 
                csv = data.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Full Historical CSV",
                    data=csv,
                    file_name=f'{ticker}_technical_data.csv',
                    mime='text/csv',
                )

            st.divider()

            # --- UI: LSTM Prediction Model + Backtesting ---
            st.subheader("🤖 AI Trend Prediction & Historical Backtest")

            with st.spinner("Training AI Model and running backtest..."):

                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))
                window_size = 60

                if len(scaled_data) < (window_size + backtest_days):
                    st.warning("Not enough historical data for backtesting. Select a longer period.")
                    next_day_price = current_price
                else:
                    train_data = scaled_data[:-backtest_days]
                    test_data = scaled_data[-(backtest_days + window_size):]

                    X_train, y_train = [], []

                    for i in range(window_size, len(train_data)):
                        X_train.append(train_data[i-window_size:i, 0])
                        y_train.append(train_data[i, 0])

                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

                    model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
                        LSTM(50),
                        Dense(1)
                    ])

                    model.compile(optimizer="adam", loss="mse")
                    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

                    X_test, y_test = [], []

                    for i in range(window_size, len(test_data)):
                        X_test.append(test_data[i-window_size:i, 0])
                        y_test.append(test_data[i, 0])

                    X_test = np.array(X_test)
                    y_test = np.array(y_test)
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                    predicted = model.predict(X_test, verbose=0)
                    predicted = scaler.inverse_transform(predicted)
                    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

                    mae = mean_absolute_error(actual, predicted)
                    rmse = np.sqrt(mean_squared_error(actual, predicted))
                    accuracy = 100 - (np.mean(np.abs((actual - predicted) / actual)) * 100)

                    st.subheader("📊 Historical Prediction Accuracy")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy %", f"{accuracy:.2f}%")
                    with col2:
                        st.metric("MAE", f"₹ {mae:.2f}")
                    with col3:
                        st.metric("RMSE", f"₹ {rmse:.2f}")

                    fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
                    test_dates = close_prices.index[-backtest_days:]

                    ax_bt.plot(test_dates, actual, label="Actual Price", linewidth=2)
                    ax_bt.plot(test_dates, predicted, label="Predicted Price", linestyle="--")

                    ax_bt.set_title(f"{ticker} Historical Backtest")
                    ax_bt.set_ylabel("Price (INR)")
                    ax_bt.legend()
                    ax_bt.grid(alpha=0.3)
                    st.pyplot(fig_bt)

                    st.divider()

                    last_60_days = scaled_data[-window_size:]
                    next_input = np.reshape(last_60_days, (1, window_size, 1))
                    next_scaled = model.predict(next_input, verbose=0)
                    next_day_price = scaler.inverse_transform(next_scaled)[0][0]

                    st.subheader("🔮 Next Trading Day Forecast")

                    p1, p2 = st.columns(2)
                    with p1:
                        st.info(f"Last Actual Close: ₹ {current_price:.2f}")
                    with p2:
                        if next_day_price > current_price:
                            st.success(f"🟢 Predicted Next Close: ₹ {next_day_price:.2f}")
                        else:
                            st.error(f"🔴 Predicted Next Close: ₹ {next_day_price:.2f}")
            st.divider()

            # --- UI: ACTIONABLE TRADING SUMMARY ---
            st.subheader("💡 Actionable Trading Summary")
            
            latest_ma5 = data['MA5'].iloc[-1]
            latest_ma10 = data['MA10'].iloc[-1]
            latest_rsi = data['RSI'].iloc[-1]
            latest_macd = data['MACD'].iloc[-1]
            
            consensus_score = 0
            
            if latest_ma5 > latest_ma10:
                ma_status = "🟢 Bullish (Short-term MA > Long-term MA)"
                consensus_score += 1
            else:
                ma_status = "🔴 Bearish (Short-term MA < Long-term MA)"
                consensus_score -= 1
                
            if latest_rsi > 70:
                rsi_status = f"🔴 Overbought at {latest_rsi:.1f} (Might drop soon)"
                consensus_score -= 1
            elif latest_rsi < 30:
                rsi_status = f"🟢 Oversold at {latest_rsi:.1f} (Good buying opportunity)"
                consensus_score += 1
            else:
                rsi_status = f"🟡 Neutral at {latest_rsi:.1f}"
                
            if latest_macd > 0:
                macd_status = "🟢 Positive Momentum"
                consensus_score += 1
            else:
                macd_status = "🔴 Negative Momentum"
                consensus_score -= 1
                
            if next_day_price > current_price:
                lstm_status = "🟢 AI Expects Price to Rise"
                consensus_score += 1
            else:
                lstm_status = "🔴 AI Expects Price to Fall"
                consensus_score -= 1

            if consensus_score >= 3:
                verdict, color = "STRONG BUY 📈", "lightgreen"
            elif consensus_score in [1, 2]:
                verdict, color = "BUY / ACCUMULATE 🛒", "lightblue"
            elif consensus_score == 0:
                verdict, color = "HOLD / NEUTRAL ⏳", "orange"
            elif consensus_score in [-1, -2]:
                verdict, color = "SELL / REDUCE 📉", "lightcoral"
            else:
                verdict, color = "STRONG SELL 🚨", "red"

            col_sum1, col_sum2 = st.columns([1, 1])
            with col_sum1:
                st.markdown("### Individual Indicators")
                st.markdown(f"- **Moving Averages:** {ma_status}")
                st.markdown(f"- **RSI (14 Days):** {rsi_status}")
                st.markdown(f"- **MACD:** {macd_status}")
                st.markdown(f"- **AI LSTM Model:** {lstm_status}")
            
            with col_sum2:
                st.markdown("### Final Consensus Verdict")
                st.markdown(
                    f"<div style='text-align: center; padding: 20px; border-radius: 10px; background-color: {color}; color: black;'>"
                    f"<h2 style='margin:0;'>{verdict}</h2>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
                
            st.caption("\n\n*⚠️ **Disclaimer**: This summary is generated algorithmically based on technical indicators and AI models. Stock markets are highly volatile and influenced by unpredictable real-world news. Do not use this as your sole basis for financial trading or investment.*")

