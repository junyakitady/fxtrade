import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to path to allow importing from other components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import fetch_data
from indicator import calculate_super_bollinger

def run_custom_backtest(
    df: pd.DataFrame,
    entry_type: int,
    exit_type: int,
    volume: int = 10000,
    initial_capital: float = 1000000
) -> dict:
    """
    スーパーボリンジャー算出済みのデータフレーム上で、指定されたエントリー条件・エグジット条件に基づき
    売買シミュレーションを実行する。
    
    エントリー条件:
    - entry_type == 1: 遅行スパン好転 & +1σ超え & バンド拡大
      LONG: Close > past_high_21 AND Close > plus_1sigma AND is_expansion
      SHORT: Close < past_low_21 AND Close < minus_1sigma AND is_expansion
    - entry_type == 2: 遅行スパン好転 & +2σ超え & バンド拡大
      LONG: Close > past_high_21 AND Close > plus_2sigma AND is_expansion
      SHORT: Close < past_low_21 AND Close < minus_2sigma AND is_expansion
      
    エグジット条件:
    - exit_type == 3: 遅行スパン陰転
      LONG Exit: Close < past_low_21
      SHORT Exit: Close > past_high_21
    - exit_type == 4: 21MA割れ
      LONG Exit: Close < center_line
      SHORT Exit: Close > center_line
    """
    if df is None or df.empty or 'plus_1sigma' not in df.columns:
        return {
            "total_trades": 0, "long_trades": 0, "short_trades": 0,
            "wins": 0, "win_rate": 0.0, "total_profit": 0.0,
            "profit_factor": 0.0, "max_dd_amount": 0.0, "max_dd_percent": 0.0,
            "current_position": 0, "trades": []
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
            # --- 新規エントリー判定 ---
            if entry_type == 1:
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
            elif entry_type == 2:
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
                    
        elif current_pos == 1:
            # --- 買いポジション決済判定 ---
            do_exit = False
            if exit_type == 3:
                if close_p < past_low:
                    do_exit = True
            elif exit_type == 4:
                if close_p < center_line:
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
            if exit_type == 3:
                if close_p > past_high:
                    do_exit = True
            elif exit_type == 4:
                if close_p > center_line:
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
    
    # プロフィットファクターの計算
    gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
    gross_loss = sum(abs(t['profit']) for t in trades if t['profit'] < 0)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float('inf') if gross_profit > 0 else 0.0
        
    # 最大ドローダウンの計算
    equity = initial_capital + df['cumulative_profit']
    peaks = equity.cummax()
    drawdowns = peaks - equity
    max_dd_amount = drawdowns.max()
    
    safe_peaks = peaks.replace(0, float('nan'))
    drawdown_rates = drawdowns / safe_peaks
    max_dd_percent = drawdown_rates.max() * 100
    if pd.isna(max_dd_percent):
        max_dd_percent = 0.0
        
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
        "trades": trades
    }

def run_custom_cross_matrix():
    print("データ取得中 (JPY=X)...")
    try:
        # オンライン取得 (エラーはそのまま例外をスローして伝播させる)
        data = fetch_data(symbol="JPY=X", period_1h="730d", period_1d="5y", raise_errors=True)
    except Exception as e:
        print("==========================================================================================")
        print("【エラー】バックテスト用データの取得に失敗しました。")
        print(f"原因: {str(e)}")
        print("==========================================================================================")
        sys.exit(1)
        
    timeframes = [("1h", "1時間足"), ("4h", "4時間足"), ("1d", "日足")]
    
    entry_conditions = [
        (1, "Entry 1: 遅行スパン好転 & +1σ超え & バンド拡大"),
        (2, "Entry 2: 遅行スパン好転 & +2σ超え & バンド拡大")
    ]
    
    exit_conditions = [
        (3, "Exit 3: 遅行スパン陰転"),
        (4, "Exit 4: 21MA割れ")
    ]
    
    results = []
    
    print("\n======================================================================================================================================================")
    print("                                   USD/JPY 3つの時間足 × 4つの条件組み合わせ (計12パターン) バックテストレポート                                    ")
    print("======================================================================================================================================================\n")
    
    for tf_code, tf_name in timeframes:
        df = data.get(tf_code)
        if df is None or df.empty:
            print(f"【警告】{tf_name} のデータが空のためスキップします。")
            continue
            
        df_ind = calculate_super_bollinger(df)
        
        print(f"■ 時間軸: {tf_name} (データ数: {len(df_ind)} 行)")
        print("-" * 150)
        print(f"{'エントリー条件 / エグジット条件':<60} | {'総取引回数 (買/売)':<20} | {'勝率':<10} | {'損益合計':<15} | {'PF':<8} | {'最大ドローダウン (額 / %)'}")
        print("-" * 150)
        
        for entry_id, entry_name in entry_conditions:
            for exit_id, exit_name in exit_conditions:
                res = run_custom_backtest(df_ind, entry_type=entry_id, exit_type=exit_id, volume=10000)
                
                trades = res['total_trades']
                l_cnt = res['long_trades']
                s_cnt = res['short_trades']
                rate = res['win_rate']
                prof = res['total_profit']
                pf = res['profit_factor']
                dd_amt = res['max_dd_amount']
                dd_pct = res['max_dd_percent']
                
                trade_str = f"{trades} 回 ({l_cnt}/{s_cnt})"
                rate_str = f"{rate:.1f} %"
                prof_str = f"{prof:+,.0f} 円"
                pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
                dd_str = f"{dd_amt:,.0f} 円 ({dd_pct:.2f}%)"
                
                comb_name = f"Entry {entry_id} × Exit {exit_id}"
                print(f"{comb_name:<60} | {trade_str:<20} | {rate_str:<10} | {prof_str:<15} | {pf_str:<8} | {dd_str}")
                
                results.append({
                    "timeframe": tf_name,
                    "entry": entry_name,
                    "exit": exit_name,
                    "trades": trade_str,
                    "win_rate": rate_str,
                    "profit": prof_str,
                    "pf": pf_str,
                    "dd": dd_str
                })
                
        print("-" * 150 + "\n")
        
    # markdown形式でサマリーを表示
    print("\n### Markdown形式サマリー表\n")
    for tf_code, tf_name in timeframes:
        print(f"#### 【{tf_name}】")
        print("| エントリー条件 | エグジット条件 | 取引回数 (買/売) | 勝率 | 損益合計 | PF | 最大ドローダウン |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        tf_results = [r for r in results if r['timeframe'] == tf_name]
        for r in tf_results:
            print(f"| {r['entry']} | {r['exit']} | {r['trades']} | {r['win_rate']} | {r['profit']} | {r['pf']} | {r['dd']} |")
        print()

if __name__ == "__main__":
    run_custom_cross_matrix()
