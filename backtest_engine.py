import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

# កំណត់ Page Config របស់ Streamlit ជាមុនសិន
st.set_page_config(page_title="Multi Crypto Analysis & Backtest", layout="wide")

# ========================================================================================
# 🛠️ PURE PANDAS/NUMPY INDICATORS (លែងប្រើ pandas_ta ដើម្បីកុំឱ្យ Crash លើ Cloud)
# ========================================================================================

def calculate_rsi_clean(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd_clean(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def calculate_adx_clean(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high.diff()
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    
    # ធានាថាបំលែងជា Series ដើម្បីកុំឱ្យជួបបញ្ហា Index ពេលប្រើ .ewm()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, min_periods=period).mean() / atr.values)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, min_periods=period).mean() / atr.values)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, min_periods=period).mean()

def calculate_supertrend_clean(df, period=10, multiplier=3.0):
    hl2 = (df['high'] + df['low']) / 2
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    
    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    direction = np.ones(len(df))
    supertrend = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if lower_basic.iloc[i] > lower_band.iloc[i-1] or df['close'].iloc[i-1] < lower_band.iloc[i-1]:
            lower_band.iloc[i] = lower_basic.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i-1]
            
        if upper_basic.iloc[i] < upper_band.iloc[i-1] or df['close'].iloc[i-1] > upper_band.iloc[i-1]:
            upper_band.iloc[i] = upper_basic.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i-1]
            
        if df['close'].iloc[i] > upper_band.iloc[i-1]:
            direction[i] = 1
        elif df['close'].iloc[i] < lower_band.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        supertrend[i] = lower_band.iloc[i] if direction[i] == 1 else upper_band.iloc[i]
        
    df['SuperTrend'] = supertrend
    df['ST_Bullish'] = [True if x == 1 else False for x in direction]
    return df

# ========================================================================================
# 📥 DATA FETCH & BACKTEST
# ========================================================================================

def fetch_historical_data(symbol="BOME/USDT", timeframe="1h", limit=500):
    # កំណត់ទម្រង់គូជួញដូរឱ្យត្រូវតាម Binance (ដកសញ្ញា / ចេញបើចាំបាច់ ប៉ុន្តែ CCXT ភាគច្រើនស្គាល់ Symbol ស្តង់ដារ)
    # ប្តូរទៅប្រើប្រាស់ Binance US Endpoint ព្រោះ Server Streamlit ស្ថិតនៅអាមេរិក
    exchange = ccxt.binanceus({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot'
        }
    })
    
    try:
        # ចំណាំ៖ Binance US អាចនឹងមិនមានគូកាក់ Meme មួយចំនួនដូច Binance.com ឡើយ
        # ប្រសិនបើវាលោត Error ថាគ្មានកាក់ BOME យើងនឹងប្រើវិធីទី២ គឺផ្លាស់ប្តូរ Base URL នៃ API វិញ
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
        
    except Exception as e:
        # ករណី Binance US គ្មានកាក់ BOME/USDT វានឹងរត់ចូលមកចំណុចនេះ 
        # យើងនឹងព្យាយាមប្រើប្រាស់ Public Proxy API របស់ Binance.com វិញម្តង
        try:
            st.warning("កំពុងសាកល្បងទាញទិន្នន័យតាមរយៈ Alternative Binance Endpoint...")
            exchange_alt = ccxt.binance({
                'enableRateLimit': True,
                'urls': {
                    'api': {
                        'public': 'https://api1.binance.com/api/v3', # ប្តូរទៅប្រើ API Cluster ផ្សេងរបស់ Binance
                    }
                }
            })
            bars = exchange_alt.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as ex:
            st.error(f"Error fetching from Binance System: {ex}")
            return None

# ========================================================================================
# 🖥️ STREAMLIT UI DISPLAY (ចំណុចដែលធ្វើឱ្យលែងចេញផ្ទាំងស)
# ========================================================================================

st.title("🚀 Multi Crypto Real-time Analysis & Backtest")

# Sidebar Settings ដូចរូបភាពចាស់របស់បង
st.sidebar.header("⚙️ Settings")
st.sidebar.subheader("Indicator Settings")
st_period = st.sidebar.slider("SuperTrend Period", 5, 30, 10)
st_multiplier = st.sidebar.slider("SuperTrend Multiplier", 1.0, 5.0, 3.0, 0.1)

# ជ្រើសរើស Coin
coin_input = st.selectbox("ជ្រើសរើស Coin:", ["BOME/USDT", "BTC/USDT", "ETH/USDT", "SOL/USDT"])

if st.button("📊 Run Analysis & Backtest"):
    with st.spinner("កំពុងទាញយកទិន្នន័យ និងគណនា..."):
        df = fetch_historical_data(symbol=coin_input, timeframe="1h", limit=500)
        
        if df is not None:
            # គណនា Indicators
            df['RSI'] = calculate_rsi_clean(df['close'], period=14)
            df = calculate_supertrend_clean(df, period=st_period, multiplier=st_multiplier)
            df['MACD_Hist'] = calculate_macd_clean(df['close'])
            df['ADX'] = calculate_adx_clean(df, period=14)
            df['Avg_Volume'] = df['volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Avg_Volume'].replace(0, np.nan)
            
            # យកទិន្នន័យចុងក្រោយបង្អស់ (Latest Row) មកបង្ហាញលើ Metrics Card
            latest = df.dropna().iloc[-1]
            prev = df.dropna().iloc[-2]
            
            # បង្កើត Columns សម្រាប់លោតកាត ដូចរូបភាពទី ១ របស់បង
            col1, col2, col3, col4, col5 = st.columns(5)
            
            price_diff = ((latest['close'] - prev['close']) / prev['close']) * 100
            col1.metric("💰 តម្លៃបច្ចុប្បន្ន", f"${latest['close']:.4f}", f"{price_diff:+.2f}%")
            
            col2.metric("📊 RSI (14)", f"{latest['RSI']:.1f}", "Normal" if 30 <= latest['RSI'] <= 70 else "Over")
            
            st_status = "🟢 Bullish" if latest['ST_Bullish'] else "🔴 Bearish"
            col3.metric("📈 SuperTrend", f"${latest['SuperTrend']:.4f}", st_status)
            
            col4.metric("📦 Volume Ratio", f"{latest['Volume_Ratio']:.2f}x")
            
            macd_status = "▲ UP" if latest['MACD_Hist'] > 0 else "▼ DOWN"
            col5.metric("⚡ MACD Hist", f"{latest['MACD_Hist']:.5f}", macd_status)
            
            # --------------------------------------------------------------------------------
            # RUN BACKTEST LOGIC SHORT SUMMARY
            # --------------------------------------------------------------------------------
            st.subheader("📊 លទ្ធផល Backtest សាកល្បង (Market Regime Filter)")
            
            balance = 1000.0
            position = 0.0
            trades = 0
            wins = 0
            
            for i in range(50, len(df)):
                row = df.iloc[i]
                c_close = row['close']
                if position == 0.0 and row['ADX'] > 22 and row['ST_Bullish'] and row['RSI'] < 70:
                    position = balance / c_close
                    balance = 0.0
                    trades += 1
                elif position > 0.0 and (not row['ST_Bullish'] or row['RSI'] > 75):
                    balance = position * c_close
                    position = 0.0
                    wins += 1 # សម្រាប់គំរូជាមូលដ្ឋាន
            
            if position > 0.0:
                balance = position * df.iloc[-1]['close']
                
            final_return = ((balance - 1000.0) / 1000.0) * 100
            
            st.success(f"💰 ប្រាក់ដើម: $1000 | 💸 ផលចំណេញសរុប: {final_return:+.2f}% | 🔄 ការជួញដូរសរុប: {trades} ដង")
            
            # បង្ហាញតារាងទិន្នន័យការពារកុំឱ្យផ្ទាំងស
            st.subheader("📋 ទិន្នន័យលម្អិត (Data Table)")
            st.dataframe(df[['date', 'open', 'high', 'low', 'close', 'RSI', 'SuperTrend', 'ADX']].tail(10))
