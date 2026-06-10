import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta  # ប្រើសម្រាប់គណនា ADX ឱ្យបានលឿន និងត្រឹមត្រូវ
from datetime import datetime

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
    """ 
    ប្រព័ន្ធ Backtesting Engine ជាមួយ Market Regime Filter (ADX)
    """
    # 1. គណនា Indicators មូលដ្ឋាន
    df['RSI'] = ta.rsi(df['close'], length=14)
    df = calculate_supertrend_clean(df, period=10, multiplier=3.0)
    
    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD_Hist'] = macd_df['MACDs_12_26_9'] # MACD Histogram
    df['Avg_Volume'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Avg_Volume'].replace(0, np.nan)
    
    # 2. បន្ថែម MARKET REGIME FILTER (ADX)
    # ta.adx នឹងផ្តល់ជួរឈរ [ADX_14, DMP_14, DMN_14]
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_df['ADX_14']
    
    # បង្កើតជួរឈរសម្រាប់ត្រួតពិនិត្យដំណើរការ Backtest
    balance = initial_balance
    position = 0.0  # ចំនួនកាក់ដែលកាន់កាប់
    entry_price = 0.0
    trades_count = 0
    winning_trades = 0
    
    trade_log = []

    # ចាប់ផ្តើម Loop ពីបារទី 30 ដើម្បីឱ្យ Indicators គណនាលំនឹងជាមុនសិន
    for i in range(30, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        close = current_row['close']
        rsi = current_row['RSI']
        st_bull = current_row['ST_Bullish']
        st = current_row['SuperTrend']
        macd_hist = current_row['MACD_Hist']
        vol_ratio = current_row['Volume_Ratio']
        adx = current_row['ADX']
        
        # ---------------------------------------------------------
        # MARKET REGIME FILTER LOGIC
        # ---------------------------------------------------------
        is_trending = adx > 22  # ទីផ្សារមាន Trend ខ្លាំង
        
        # កំណត់លក្ខខណ្ឌទិញ/លក់ ផ្អែកលើស្ថានភាពទីផ្សារ
        if is_trending:
            # 📈 យុទ្ធសាស្ត្រពេលទីផ្សារមាន Trend (Trend-Following - ដូចកូដចាស់របស់អ្នក)
            buy_signal = (st_bull and close > st and 50 < rsi < 70 and macd_hist > 0 and vol_ratio > 1.1)
            sell_signal = (not st_bull or close < st or rsi > 75)
        else:
            # 🔄 យុទ្ធសាស្ត្រពេលទីផ្សារ Sideway (Mean-Reversion - ទិញទាប លក់ខ្ពស់)
            buy_signal = (rsi < 32 and macd_hist > prev_row['MACD_Hist'])  # ទិញពេល Oversold ហើយកម្លាំងធ្លាក់ចុះថយ
            sell_signal = (rsi > 68)  # លក់លឿនមុនពេលវាទម្លាក់ក្បាលចុះ
            
        # ---------------------------------------------------------
        # EXECUTION LOGIC (ទិញ និង លក់)
        # ---------------------------------------------------------
        # លក្ខខណ្ឌចូលទិញ (Buy/Long)
        if position == 0.0 and buy_signal:
            # ប្រើប្រាស់ដើមទុនទាំងអស់ (All-in) សម្រាប់តេស្ត
            position = (balance * (1 - fee_rate)) / close
            entry_price = close
            balance = 0.0
            trades_count += 1
            trade_log.append(f"🟢 [BUY]  Date: {current_row['date']} | Price: ${close:.4f} | ADX: {adx:.1f} ({'Trending' if is_trending else 'Sideway'})")
            
        # លក្ខខណ្ឌលក់ចេញ (Sell/Close Position)
        elif position > 0.0 and sell_signal:
            balance = (position * close) * (1 - fee_rate)
            pnl_pct = ((close - entry_price) / entry_price) * 100
            
            if pnl_pct > 0:
                winning_trades += 1
                
            trade_log.append(f"🔴 [SELL] Date: {current_row['date']} | Price: ${close:.4f} | PnL: {pnl_pct:+.2f}%")
            position = 0.0
            entry_price = 0.0

    # បិទ Position ចុងក្រោយបើមិនទាន់បានលក់នៅចុងបញ្ចប់នៃទិន្នន័យ
    if position > 0.0:
        balance = (position * df.iloc[-1]['close']) * (1 - fee_rate)
        pnl_pct = ((df.iloc[-1]['close'] - entry_price) / entry_price) * 100
        if pnl_pct > 0: winning_trades += 1
        trade_log.append(f"🔴 [FORCE CLOSE] Price: ${df.iloc[-1]['close']:.4f} | PnL: {pnl_pct:+.2f}%")

    # 3. គណនាលទ្ធផលសរុប (Performance Metrics)
    final_return = ((balance - initial_balance) / initial_balance) * 100
    win_rate = (winning_trades / trades_count * 100) if trades_count > 0 else 0
    
    return final_return, trades_count, win_rate, trade_log

def calculate_supertrend_clean(df, period=10, multiplier=3.0):
    """ មុខងារគណនា SuperTrend ដែលមានលំនឹងខ្ពស់ និងគ្មាន Bug NaN """
    hl2 = (df['high'] + df['low']) / 2
    atr = ta.atr(df['high'], df['low'], df['close'], length=period)
    
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
# 🚀 ដំណើរការតេស្តសាកល្បង (Execution)
# ========================================================================================
if __name__ == "__main__":
    coin = "SOL/USDT"  # អាចប្តូរជា BTC/USDT, ETH/USDT, BOME/USDT
    print(f"📥 កំពុងទាញយកទិន្នន័យដើម្បីធ្វើ Backtest លើកាក់ {coin}...")
    
    # ទាញយកទិន្នន័យ ១០០០ ម៉ោងចុងក្រោយ (ប្រហែល ៤១ ថ្ងៃ)
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
        
        # បង្ហាញប្រវត្តិ Trade ៥ ដងចុងក្រោយបង្អស់
        print("\n📋 ប្រវត្តិការ Trade ចុងក្រោយមួយចំនួន៖")
        for log in logs[-6:]:
            print(log)