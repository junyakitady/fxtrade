import pandas as pd

def run_backtest(df: pd.DataFrame, volume: int = 10000) -> dict:
    """
    スーパーボリンジャー算出済みのデータフレーム上で売買シミュレーションを実行する。
    
    前提:
    - 最大1ポジションの単利運用（ロング・ショート両対応）
    - 買いエントリー (AND): 終値 > 21期間前高値 ＆ 終値 > +1σ
    - 買い決済 (AND): 終値 < 21期間前安値 ＆ 終値 < +1σ
    - 売りエントリー (AND): 終値 < 21期間前安値 ＆ 終値 < -1σ
    - 売り決済 (AND): 終値 > 21期間前高値 ＆ 終値 > -1σ
    - コスト(スプレッド・手数料)なし
    
    戻り値の内訳:
    - total_trades: 買い＋売りの合算取引回数
    - long_trades: 買いトレードの回数
    - short_trades: 売りトレードの回数
    """
    if df is None or df.empty or 'plus_1sigma' not in df.columns:
        return {
            "total_trades": 0, "long_trades": 0, "short_trades": 0,
            "wins": 0, "win_rate": 0.0, "total_profit": 0.0,
            "current_position": 0, "df_result": pd.DataFrame(), "trades": []
        }
        
    df = df.copy()
    df['signal'] = 0
    df['position'] = 0
    df['trade_profit'] = 0.0
    df['cumulative_profit'] = 0.0
    
    current_pos = 0
    entry_price = 0.0
    entry_time = None
    
    trades = []
    cumulative_profit = 0.0
    
    col_sig = df.columns.get_loc('signal')
    col_pos = df.columns.get_loc('position')
    col_prof = df.columns.get_loc('trade_profit')
    col_cum = df.columns.get_loc('cumulative_profit')
    
    for i in range(len(df)):
        row = df.iloc[i]
        idx = df.index[i]
        
        if pd.isna(row['plus_1sigma']) or pd.isna(row['past_high_21']):
            continue
            
        close_p = row['Close']
        plus_1s = row['plus_1sigma']
        minus_1s = row['minus_1sigma']
        past_high = row['past_high_21']
        past_low = row['past_low_21']
        
        if current_pos == 0:
            if close_p > past_high and close_p > plus_1s:
                current_pos = 1
                entry_price = close_p
                entry_time = idx
                df.iat[i, col_sig] = 1
            elif close_p < past_low and close_p < minus_1s:
                current_pos = -1
                entry_price = close_p
                entry_time = idx
                df.iat[i, col_sig] = -1
        elif current_pos == 1:
            if close_p < past_low and close_p < plus_1s:
                current_pos = 0
                exit_price = close_p
                profit = (exit_price - entry_price) * volume
                cumulative_profit += profit
                trades.append({
                    "type": "LONG", "entry_time": entry_time, "exit_time": idx,
                    "entry_price": entry_price, "exit_price": exit_price, "profit": profit
                })
                df.iat[i, col_sig] = 2
                df.iat[i, col_prof] = profit
        elif current_pos == -1:
            if close_p > past_high and close_p > minus_1s:
                current_pos = 0
                exit_price = close_p
                profit = (entry_price - exit_price) * volume
                cumulative_profit += profit
                trades.append({
                    "type": "SHORT", "entry_time": entry_time, "exit_time": idx,
                    "entry_price": entry_price, "exit_price": exit_price, "profit": profit
                })
                df.iat[i, col_sig] = -2
                df.iat[i, col_prof] = profit
                
        df.iat[i, col_pos] = current_pos
        df.iat[i, col_cum] = cumulative_profit
        
    total_trades = len(trades)
    long_trades = sum(1 for t in trades if t['type'] == 'LONG')
    short_trades = sum(1 for t in trades if t['type'] == 'SHORT')
    
    wins = sum(1 for t in trades if t['profit'] > 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_profit = sum(t['profit'] for t in trades)
    
    return {
        "total_trades": total_trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "wins": wins,
        "win_rate": win_rate,
        "total_profit": total_profit,
        "current_position": current_pos,
        "df_result": df,
        "trades": trades
    }
