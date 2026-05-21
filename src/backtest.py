import pandas as pd

def run_backtest(df: pd.DataFrame, timeframe: str = "1h", volume: int = 10000, initial_capital: float = 1000000) -> dict:
    """
    スーパーボリンジャー算出済みのデータフレーム上で売買シミュレーションを実行する。
    
    適用ロジック: 時間軸別の「平均利益トップモデル」（グリッドサーチ厳選設定）
    - 最大1ポジションの単利運用（ロング・ショート両対応）
    
    【1時間足 (1h)】
    - 新規エントリー: 遅行スパン好転 ＆ 終値 > +2σ ＆ バンド拡大
    - 決済: 遅行スパン逆転のみ
    【4時間足 (4h)】
    - 新規エントリー: 遅行スパン好転 ＆ 終値 > +1σ
    - 決済: 終値 < +1σ のみ
    【日足 (1d)】
    - 新規エントリー: 遅行スパン好転 ＆ 終値 > +1σ ＆ バンド拡大
    - 決済: 終値 < センターライン(21SMA) のみ
    """
    if df is None or df.empty or 'plus_1sigma' not in df.columns:
        return {
            "total_trades": 0, "long_trades": 0, "short_trades": 0,
            "wins": 0, "win_rate": 0.0, "total_profit": 0.0,
            "profit_factor": 0.0, "max_dd_amount": 0.0, "max_dd_percent": 0.0,
            "current_position": 0, "df_result": pd.DataFrame(), "trades": []
        }
        
    df = df.copy()
    
    # エクスパンション判定用の差分列を作成
    df['m2s_diff'] = df['minus_2sigma'].diff()
    df['p2s_diff'] = df['plus_2sigma'].diff()
    
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
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        idx = df.index[i]
        
        if pd.isna(row['plus_1sigma']) or pd.isna(row['past_high_21']):
            continue
            
        close_p = row['Close']
        plus_1s = row['plus_1sigma']
        minus_1s = row['minus_1sigma']
        plus_2s = row['plus_2sigma']
        minus_2s = row['minus_2sigma']
        center_line = row['center_line']
        past_high = row['past_high_21']
        past_low = row['past_low_21']
        
        # 本物のエクスパンション（バンド幅の拡大）判定: -2σが下向き かつ +2σが上向き
        m2s_down = row['m2s_diff'] < 0
        p2s_up = row['p2s_diff'] > 0
        is_expansion = m2s_down and p2s_up
        
        if current_pos == 0:
            # --- 新規エントリー判定 (時間軸別の特化モデル) ---
            if timeframe == "1h":
                if close_p > past_high and close_p > plus_2s and is_expansion:
                    current_pos = 1
                    entry_price = close_p
                    entry_time = idx
                    df.iat[i, col_sig] = 1
                elif close_p < past_low and close_p < minus_2s and is_expansion:
                    current_pos = -1
                    entry_price = close_p
                    entry_time = idx
                    df.iat[i, col_sig] = -1
            elif timeframe == "4h":
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
            elif timeframe == "1d":
                if close_p > past_high and close_p > plus_1s and is_expansion:
                    current_pos = 1
                    entry_price = close_p
                    entry_time = idx
                    df.iat[i, col_sig] = 1
                elif close_p < past_low and close_p < minus_1s and is_expansion:
                    current_pos = -1
                    entry_price = close_p
                    entry_time = idx
                    df.iat[i, col_sig] = -1
                    
        elif current_pos == 1:
            # --- 買いポジション決済判定 ---
            do_exit = False
            if timeframe == "1h" and close_p < past_low:
                do_exit = True
            elif timeframe == "4h" and close_p < plus_1s:
                do_exit = True
            elif timeframe == "1d" and close_p < center_line:
                do_exit = True
                
            if do_exit:
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
            # --- 売りポジション決済判定 ---
            do_exit = False
            if timeframe == "1h" and close_p > past_high:
                do_exit = True
            elif timeframe == "4h" and close_p > minus_1s:
                do_exit = True
            elif timeframe == "1d" and close_p > center_line:
                do_exit = True
                
            if do_exit:
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
    
    # プロフィットファクター（PF）の計算
    gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
    gross_loss = sum(abs(t['profit']) for t in trades if t['profit'] < 0)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float('inf') if gross_profit > 0 else 0.0
        
    # 最大ドローダウン（Max DD）の計算
    equity = initial_capital + df['cumulative_profit']
    peaks = equity.cummax()
    drawdowns = peaks - equity
    max_dd_amount = drawdowns.max()
    
    safe_peaks = peaks.replace(0, float('nan'))
    drawdown_rates = drawdowns / safe_peaks
    max_dd_percent = drawdown_rates.max() * 100
    if pd.isna(max_dd_percent):
        max_dd_percent = 0.0
        
    # 現在の保有ポジションに関する追加情報を計算
    current_entry_price = 0.0
    current_entry_time = None
    current_unrealized_profit = 0.0
    
    if current_pos != 0 and not df.empty:
        current_entry_price = entry_price
        current_entry_time = entry_time
        latest_close = df['Close'].iloc[-1]
        # 買ポジション(1): (Close - Entry) * Vol
        # 売ポジション(-1): (Entry - Close) * Vol  -> (Close - Entry) * Vol * (-1) と同等
        current_unrealized_profit = (latest_close - entry_price) * volume * current_pos
        
    return {
        "total_trades": total_trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "wins": wins,
        "win_rate": win_rate,
        "total_profit": total_profit,
        "profit_factor": profit_factor,
        "max_dd_amount": max_dd_amount,
        "max_dd_percent": max_dd_percent,
        "current_position": current_pos,
        "current_entry_price": current_entry_price,
        "current_entry_time": current_entry_time,
        "current_unrealized_profit": current_unrealized_profit,
        "df_result": df,
        "trades": trades
    }
