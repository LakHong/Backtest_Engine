import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

# ========================================================================================
# 🛠️ PURE PANDAS/NUMPY INDICATORS (ជំនួស pandas_ta ដើម្បីកុំឱ្យលោត Error លើ Cloud)
# ========================================================================================

def calculate_rsi_clean(series, period=14):
    """ គណនា RSI ដោយប្រើប្រាស់វិធីសាស្ត្រ Wilder's Moving Average """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd_clean(series, fast=12, slow=26, signal=9):
    """ គណនា MACD Line, Signal Line និង Histogram """
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_hist

def calculate_adx_clean(df, period=14):
    """ គណនា ADX (Average Directional Index) """
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
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, min_periods=period).mean() / atr.values)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, min_periods=period).mean() / atr.values)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx

def calculate_atr_clean(df, period=10):
    """ គណនា ATR សម្រាប់យកទៅប្រើប្រាស់ក្នុង SuperTrend """
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()

def calculate_supertrend_clean(df, period=10, multiplier=3.0):
    """ មុខងារគណនា SuperTrend ដែលមានលំនឹងខ្ពស់ និងគ្មាន Bug NaN """
    hl2 = (df['high'] + df['low']) / 2
    atr = calculate_atr_clean(df, period=period)
    
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
# 📈 CORE BACKTEST ENGINE
# ========================================================================================

def fetch_historical_data(symbol="BTC/USDT", timeframe="1h", limit=1000):
    """ ទាញយកទិន្នន័យអតីតកាលចំនួនច្រើនបារសម្រាប់ធ្វើ Backtest """
    exchange = ccxt.binance()
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def run_backtest(df, initial_balance=1000.0, fee_rate=0.001):
    """ ប្រព័ន្ធ Backtesting Engine ជាមួយ Market Regime Filter (ADX) """
    # 1. គណនា Indicators មូលដ្ឋាន (ប្តូរមកប្រើមុខងារ Clean ទាំងអស់)
    df['RSI'] = calculate_rsi_clean(df['close'], period=14)
    df = calculate_supertrend_clean(df, period=10, multiplier=3.0)
    df['MACD_Hist'] = calculate_macd_clean(df['close'], fast=12, slow=26, signal=9)
    
    df['Avg_Volume'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Avg_Volume'].replace(0, np.nan)
    
    # 2. បន្ថែម MARKET REGIME FILTER (ADX)
    df['ADX'] = calculate_adx_clean(df, period=14)
    
    balance = initial_balance
    position = 0.0  
    entry_price = 0.0
    trades_count = 0
    winning_trades = 0
    trade_log = []

    # បង្កើនជួរ Loop ទៅ ៥០ បារ ដើម្បីឱ្យទិន្នន័យគណនា EMA/EWM មានលំនឹងត្រឹមត្រូវមិនលោត NaN
    for i in range(50, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        close = current_row['close']
        rsi = current_row['RSI']
        st_bull = current_row['ST_Bullish']
        st = current_row['SuperTrend']
        macd_hist = current_row['MACD_Hist']
        vol_ratio = current_row['Volume_Ratio']
        adx = current_row['ADX']
        
        # MARKET REGIME FILTER LOGIC
        is_trending = adx > 22  
        
        if is_trending:
            # 📈 យុទ្ធសាស្ត្រពេលទីផ្សារមាន Trend (Trend-Following)
            buy_signal = (st_bull and close > st and 50 < rsi < 70 and macd_hist > 0 and vol_ratio > 1.1)
            sell_signal = (not st_bull or close < st or rsi > 75)
        else:
            # 🔄 យុទ្ធសាស្ត្រពេលទីផ្សារ Sideway (Mean-Reversion)
            buy_signal = (rsi < 32 and macd_hist > prev_row['MACD_Hist'])  
            sell_signal = (rsi > 68)  
            
        # EXECUTION LOGIC (ទិញ និង លក់)
        if position == 0.0 and buy_signal:
            position = (balance * (1 - fee_rate)) / close
            entry_price = close
            balance = 0.0
            trades_count += 1
            trade_log.append(f"🟢 [BUY]  Date: {current_row['date']} | Price: ${close:.4f} | ADX: {adx:.1f} ({'Trending' if is_trending else 'Sideway'})")
            
        elif position > 0.0 and sell_signal:
            balance = (position * close) * (1 - fee_rate)
            pnl_pct = ((close - entry_price) / entry_price) * 100
            if pnl_pct > 0:
                winning_trades += 1
                
            trade_log.append(f"🔴 [SELL] Date: {current_row['date']} | Price: ${close:.4f} | PnL: {pnl_pct:+.2f}%")
            position = 0.0
            entry_price = 0.0

    if position > 0.0:
        balance = (position * df.iloc[-1]['close']) * (1 - fee_rate)
        pnl_pct = ((df.iloc[-1]['close'] - entry_price) / entry_price) * 100
        if pnl_pct > 0: winning_trades += 1
        trade_log.append(f"🔴 [FORCE CLOSE] Price: ${df.iloc[-1]['close']:.4f} | PnL: {pnl_pct:+.2f}%")

    final_return = ((balance - initial_balance) / initial_balance) * 100
    win_rate = (winning_trades / trades_count * 100) if trades_count > 0 else 0
    
    return final_return, trades_count, win_rate, trade_log

# ========================================================================================
# 🚀 RUN TEST
# ========================================================================================
if __name__ == "__main__":
    coin = "SOL/USDT"  
    print(f"📥 កំពុងទាញយកទិន្នន័យដើម្បីធ្វើ Backtest លើកាក់ {coin}...")
    
    historical_df = fetch_historical_data(symbol=coin, timeframe="1h", limit=1000)
    
    if historical_df is not None:
        pnl, total_trades, wr, logs = run_backtest(historical_df, initial_balance=1000.0)
        
        print("\n" + "="*50)
        print(f"📊 លទ្ធផល BACKTESTING SUMMARY ({coin})")
        print("="*50)
        print(f"💰 ប្រាក់ដើមដំបូង: $1000.00")
        print(f"📈 ផលចំណេញសរុប (Total Return): {pnl:+.2f}%")
        print(f"🔄 ចំនួនដងដែលបាន Trade សរុប: {total_trades} ដង")
        print(f"🎯 ភាគរយឈ្នះ (Win Rate): {wr:.2f}%")
        print("="*50)
        
        print("\n📋 ប្រវត្តិការ Trade ចុងក្រោយមួយចំនួន៖")
        for log in logs[-6:]:
            print(log)
