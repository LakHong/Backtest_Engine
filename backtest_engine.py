import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

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
    """
    គណនា SuperTrend ធានាការបំពេញតម្លៃទិន្នន័យចុងក្រោយ (លែងចេញ N/A លើ Cloud)
    """
    df = df.reset_index(drop=True)
    
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    
    hl2 = (high + low) / 2
    
    # គណនា True Range (TR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # គណនា ATR ដោយប្រើសមីការ Wilder's Smoothing (ដូច TradingView 100%)
    atr = np.zeros(len(df))
    if len(df) >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(df)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    
    upper_band = np.copy(upper_basic)
    lower_band = np.copy(lower_basic)
    direction = np.ones(len(df))
    supertrend = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if atr[i] == 0:
            continue
        
        # Lower Band
        if lower_basic[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
            lower_band[i] = lower_basic[i]
        else:
            lower_band[i] = lower_band[i-1]
            
        # Upper Band
        if upper_basic[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
            upper_band[i] = upper_basic[i]
        else:
            upper_band[i] = upper_band[i-1]
            
        # Direction
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
        
    df['SuperTrend'] = supertrend
    df['ST_Bullish'] = [True if x == 1 else False for x in direction]
    
    # 🛠️ ដំណោះស្រាយពិសេស៖ បើសិនជាជួរចុងក្រោយបង្អស់ជាប់ None ត្រូវទាញតម្លៃមុននោះមកជំនួសភ្លាម
    df['SuperTrend'] = df['SuperTrend'].replace(0, np.nan).ffill().bfill()
    
    return df

# ========================================================================================
# 📥 DATA FETCH & BACKTEST
# ========================================================================================

def fetch_historical_data(symbol="BTC/USDT", timeframe="1h", limit=500):
    """
    ទាញយកទិន្នន័យពី Yahoo Finance ជំនួសវិញ ដើម្បីគេចពីការ Block IP 403/451 របស់ Exchange លើ Cloud
    """
    # 🔄 បំលែងទម្រង់ Symbol ពីស្តង់ដារ Exchange (ដូចជា BTC/USDT) ទៅជាទម្រង់របស់ Yahoo Finance (BTC-USD)
    yf_symbol = symbol.replace("/USDT", "-USD").replace("USDT", "-USD")
    
    # កំណត់ទម្រង់ Interval ឱ្យត្រូវគ្នា
    # សម្រាប់ Yahoo Finance: 1h, 1d, 5m (ចំណាំ៖ Interval 1h អាចទាញថយក្រោយបានត្រឹម ៧៣០ថ្ងៃប៉ុណ្ណោះ ដែលវាគ្រប់គ្រាន់សម្រាប់ 500 bars)
    yf_interval = "1h"
    if timeframe == "1d":
        yf_interval = "1d"
    
    try:
        # ហៅទាញទិន្នន័យតាមរយៈ yfinanceTicker
        ticker = yf.Ticker(yf_symbol)
        
        # ទាញយកទិន្នន័យតាមចំនួន Limit (សន្មតយករយៈពេល ២ ខែថយក្រោយសម្រាប់ 1h ល្មមបាន ៥០០ Bars)
        period_str = "60d" if yf_interval == "1h" else "max"
        df_yf = ticker.history(period=period_str, interval=yf_interval)
        
        if df_yf.empty:
            st.error(f"⚠️ មិនមានទិន្នន័យសម្រាប់គូកាក់ {yf_symbol} ឡើយនៅលើ Yahoo Finance។")
            return None
            
        # រៀបចំទម្រង់ DataFrame ឡើងវិញឱ្យត្រូវគ្នាជាមួយកូដគណនា Indicators ចាស់របស់អ្នកទាំងស្រុង
        df = df_yf.reset_index()
        
        # កែឈ្មោះជួរឈរ (Columns) ឱ្យទៅជាអក្សរតូចដើម្បីត្រូវជាមួយកូដចាស់
        df = df.rename(columns={
            'Datetime': 'date',
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # តម្រៀបយកតែចំនួន Limit ដែលបងចង់បានចុងក្រោយគេ (ឧទាហរណ៍៖ ៥០០ Bars)
        df = df.tail(limit).reset_index(drop=True)
        return df
        
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance Layer: {e}")
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
        
        if df is not None and not df.empty:
            # គណនា Indicators
            df['RSI'] = calculate_rsi_clean(df['close'], period=14)
            df = calculate_supertrend_clean(df, period=st_period, multiplier=st_multiplier)
            df['MACD_Hist'] = calculate_macd_clean(df['close'])
            df['ADX'] = calculate_adx_clean(df, period=14)
            
            # 🛠️ ផ្នែកកែសម្រួលថ្មី៖ គណនា Volume Ratio បែប Z-Score Standardized (ត្រូវទាំង BTC និង ETH)
            df['Vol_Mean'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['Vol_Std'] = df['volume'].rolling(window=20, min_periods=1).std()
            
            # គណនា Z-Score ដើម្បីវាស់សម្ពាធទំហំជួញដូរ (បូក 1e-8 ការពារកំហុសចែកនឹងសូន្យ)
            df['Vol_Z_Score'] = (df['volume'] - df['Vol_Mean']) / (df['Vol_Std'] + 1e-8)
            
            # បំប្លែងមកជា Ratio Multiplier ឱ្យនៅជុំវិញកម្រិត ១.០០x ដូចបន្ទាត់ Volume MA លើ TradingView
            df['Volume_Ratio'] = 1.0 + (df['Vol_Z_Score'] * 0.2)
            
            # លីមីតជួរតម្លៃលទ្ធផលចុងក្រោយដើម្បីកុំឱ្យ UI លោតខ្លាំងពេក
            df['Volume_Ratio'] = df['Volume_Ratio'].clip(lower=0.1, upper=2.5)
            
            # 🔄 ដំណោះស្រាយដាច់ស្រឡះ៖ មិនប្រើ .dropna() លើ DataFrame ទាំងមូលឡើយ 
            # ដើម្បីរក្សាជួរចុងក្រោយបង្អស់ (Latest Row) ឱ្យនៅដដែលទោះជាមាន NaN ក្នុង Indicator ខ្លះក៏ដោយ
            if len(df) >= 2:
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                
                # បង្កើត Columns ចំនួន ៦ ឱ្យត្រូវតាម Layout របស់បង
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                # កាតទី ១៖ តម្លៃបច្ចុប្បន្ន
                price_diff = ((latest['close'] - prev['close']) / prev['close']) * 100 if prev['close'] != 0 else 0
                col1.metric("💰 តម្លៃបច្ចុប្បន្ន", f"${latest['close']:.4f}", f"{price_diff:+.2f}%")
                
                # កាតទី ២៖ RSI (ករណីតម្លៃ NaN ឱ្យបង្ហាញ "N/A")
                rsi_val = f"{latest['RSI']:.1f}" if not np.isnan(latest['RSI']) else "N/A"
                col2.metric("📊 RSI (14)", rsi_val, "Normal" if (not np.isnan(latest['RSI']) and 30 <= latest['RSI'] <= 70) else "Over")
                
                # កាតទី ៣៖ SuperTrend
                if not np.isnan(latest['SuperTrend']) and latest['SuperTrend'] != 0:
                    st_status = "🟢 Bullish" if latest.get('ST_Bullish', True) else "🔴 Bearish"
                    col3.metric("📈 SuperTrend", f"${latest['SuperTrend']:.4f}", st_status)
                else:
                    # បើសិនជាជួរចុងក្រោយបង្អស់ NaN យើងទាញយកតម្លៃពីជួរមុននោះបន្តិចដែលមិនមែនជា NaN
                    valid_st = df['SuperTrend'].dropna()
                    if not valid_st.empty:
                        last_valid_st = valid_st.iloc[-1]
                        st_status = "🟢 Bullish" if df['ST_Bullish'].loc[valid_st.index[-1]] else "🔴 Bearish"
                        col3.metric("📈 SuperTrend", f"${last_valid_st:.4f}", st_status)
                    else:
                        col3.metric("📈 SuperTrend", "N/A", "Unknown")
                
                # កាតទី ៤៖ Volume Ratio
                v_ratio = f"{latest['Volume_Ratio']:.2f}x" if not np.isnan(latest['Volume_Ratio']) else "N/A"
                col4.metric("📦 Volume Ratio", v_ratio)
                
                # កាតទី ៥៖ MACD Histogram
                if not np.isnan(latest['MACD_Hist']):
                    macd_status = "▲ UP" if latest['MACD_Hist'] > 0 else "▼ DOWN"
                    col5.metric("⚡ MACD Hist", f"{latest['MACD_Hist']:.5f}", macd_status)
                else:
                    col5.metric("⚡ MACD Hist", "N/A", "Unknown")
                
                # កាតទី ៦៖ Signal
                current_signal = "NEUTRAL"
                if latest['ADX'] > 22 and latest.get('ST_Bullish', False) and latest['RSI'] < 70:
                    current_signal = "BUY (Trend)"
                elif latest['RSI'] < 30:
                    current_signal = "BUY (Oversold)"
                col6.metric("🎯 Signal", current_signal, "Active")
                
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
                    if position == 0.0 and row['ADX'] > 22 and row.get('ST_Bullish', False) and row['RSI'] < 70:
                        position = balance / c_close
                        balance = 0.0
                        trades += 1
                    elif position > 0.0 and (not row.get('ST_Bullish', True) or row['RSI'] > 75):
                        balance = position * c_close
                        position = 0.0
                        wins += 1
                
                if position > 0.0:
                    balance = position * df.iloc[-1]['close']
                    
                final_return = ((balance - 1000.0) / 1000.0) * 100
                st.success(f"💰 ប្រាក់ដើម: $1000 | 💸 ផលចំណេញសរុប: {final_return:+.2f}% | 🔄 ការជួញដូរសរុប: {trades} ដង")
                
                # បង្ហាញតារាងទិន្នន័យការពារកុំឱ្យផ្ទាំងស
                st.subheader("📋 ទិន្នន័យលម្អិត (Data Table)")
                st.dataframe(df[['date', 'open', 'high', 'low', 'close', 'RSI', 'SuperTrend', 'ADX']].tail(10))
            else:
                st.error("❌ ទិន្នន័យក្នុង DataFrame តិចជាង ២ ជួរ មិនអាចគណនាបានឡើយ។")
        else:
            st.error("❌ មិនអាចទាញយកទិន្នន័យបានទេ! សូមពិនិត្យមើលការភ្ជាប់ Network ឬប្រព័ន្ធទាញទិន្នន័យម្តងទៀត។")
