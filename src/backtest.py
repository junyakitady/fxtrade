import pandas as pd
import numpy as np

def run_backtest(df: pd.DataFrame, entry_strat: str = None, exit_strat: str = None, lagging_type: str = None, timeframe: str = None, volume: int = 10000, initial_capital: float = 1000000.0, lot_size: int = None) -> dict:
    """
    スーパーボリンジャー算出済みのデータフレーム上で売買シミュレーションを実行する。
    本番UIおよび一括検証の共通バックテストシミュレータ。
    
    タイムフレーム、エントリー・決済パラメータ、遅行スパン基準に基づいた汎用シミュレーションを実行。
    翌足始値約定モデル（厳密版）＋ 21MAエグジット時の21MA同方向エントリーフィルター付き。
    """
    if lot_size is not None:
        volume = lot_size

    if df is None or df.empty or 'plus_1sigma' not in df.columns:
        return {
            "total_trades": 0, "long_trades": 0, "short_trades": 0,
            "wins": 0, "win_rate": 0.0, "total_profit": 0.0,
            "profit_factor": 0.0, "max_dd_amount": 0.0, "max_dd_percent": 0.0,
            "current_position": 0, "df_result": pd.DataFrame(), "trades": []
        }
        
    # UI側の引数(timeframe文字列)から各種パラメータへ自動マッピング(後方互換性)
    if timeframe is not None:
        if timeframe == "1h_e6_ex2":
            entry_strat = "E6"
            exit_strat = "EX2"
            lagging_type = "high_low"
        elif timeframe == "1h_e1_ex3":
            entry_strat = "E1"
            exit_strat = "EX3"
            lagging_type = "high_low"
        elif timeframe == "4h_e1_ex3":
            entry_strat = "E1"
            exit_strat = "EX3"
            lagging_type = "high_low"
        elif timeframe == "1d_e2_ex5":
            entry_strat = "E2"
            exit_strat = "EX5"
            lagging_type = "high_low"
        elif timeframe == "goog_e4_ex5":
            entry_strat = "E4"
            exit_strat = "EX5"
            lagging_type = "close"
        # 互換用 (旧タイムフレームキー)
        elif timeframe == "1h":
            entry_strat = "E6"
            exit_strat = "EX1"
            lagging_type = "high_low"
        elif timeframe == "4h":
            entry_strat = "E6"
            exit_strat = "EX1"
            lagging_type = "high_low"
        elif timeframe == "1d":
            entry_strat = "E2"
            exit_strat = "EX5"
            lagging_type = "high_low"
            
    # パラメータ未指定時のデフォルトフォールバック
    if entry_strat is None:
        entry_strat = "E1"
    if exit_strat is None:
        exit_strat = "EX1"
    if lagging_type is None:
        lagging_type = "close"
        
    df = df.copy()
    
    # シグナル記録用カラム
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
    
    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        idx = df.index[i]
        
        next_row = df.iloc[i + 1]
        next_idx = df.index[i + 1]
        
        # 必要なカラムの存在チェックおよびNaNチェック
        if pd.isna(row['plus_1sigma']) or pd.isna(row['past_high_21']) or pd.isna(row.get('past_close_21')):
            df.iat[i + 1, col_pos] = current_pos
            df.iat[i + 1, col_cum] = cumulative_profit
            continue
            
        close_p = row['Close']
        center_line = row['center_line']
        
        # 遅行スパン判定（終値基準 / 高安基準の解決）
        if lagging_type == 'close':
            is_lagging_bull = row['Close'] > row['Close_21_ago']
            is_lagging_bear = row['Close'] < row['Close_21_ago']
        else: # 'high_low'
            is_lagging_bull = row['Close'] > row['High_21_ago']
            is_lagging_bear = row['Close'] < row['Low_21_ago']
            
        sig_to_set = 0
        profit_to_set = 0.0
        
        if current_pos == 0:
            # --- 新規エントリー判定 ---
            is_long_entry = False
            is_short_entry = False
            
            # 【ロングエントリー判定】
            if is_lagging_bull:
                # エクスパンション判定なし (E1-E3)
                if entry_strat == 'E1':
                    is_long_entry = True
                elif entry_strat == 'E2' and row['Close_gt_plus1']:
                    is_long_entry = True
                elif entry_strat == 'E3' and row['Close_gt_plus2']:
                    is_long_entry = True
                
                # エクスパンション判定あり (E4-E6)
                if row['Expansion']:
                    if entry_strat == 'E4':
                        is_long_entry = True
                    elif entry_strat == 'E5' and row['Close_gt_plus1']:
                        is_long_entry = True
                    elif entry_strat == 'E6' and row['Close_gt_plus2']:
                        is_long_entry = True
                        
            # 【ショートエントリー判定】
            if is_lagging_bear:
                # エクスパンション判定なし (E1-E3)
                if entry_strat == 'E1':
                    is_short_entry = True
                elif entry_strat == 'E2' and row['Close_lt_minus1']:
                    is_short_entry = True
                elif entry_strat == 'E3' and row['Close_lt_minus2']:
                    is_short_entry = True
                
                # エクスパンション判定あり (E4-E6)
                if row['Expansion']:
                    if entry_strat == 'E4':
                        is_short_entry = True
                    elif entry_strat == 'E5' and row['Close_lt_minus1']:
                        is_short_entry = True
                    elif entry_strat == 'E6' and row['Close_lt_minus2']:
                        is_short_entry = True
                        
            # 21MAフィルターの適用 (決済条件に21MA中心線を含む EX3, EX5, EX6, EX7 が対象)
            if exit_strat in ['EX3', 'EX5', 'EX6', 'EX7']:
                if is_long_entry and not (close_p > center_line):
                    is_long_entry = False
                if is_short_entry and not (close_p < center_line):
                    is_short_entry = False
                    
            # ポジション確定処理（翌足の始値で約定）
            if is_long_entry:
                current_pos = 1
                entry_price = next_row['Open']
                entry_time = next_idx
                sig_to_set = 1
            elif is_short_entry:
                current_pos = -1
                entry_price = next_row['Open']
                entry_time = next_idx
                sig_to_set = -1
                
        elif current_pos == 1:
            # --- 買いポジション決済判定 ---
            do_exit = False
            
            c5 = is_lagging_bear                 # 遅行スパン陰転
            c6 = row['Close_lt_plus1']           # 終値が+1σを割り込む
            c7 = row['Close_lt_21MA']            # 終値が21MA(センターライン)を割り込む
            
            if exit_strat == 'EX1' and c5:
                do_exit = True
            elif exit_strat == 'EX2' and c6:
                do_exit = True
            elif exit_strat == 'EX3' and c7:
                do_exit = True
            elif exit_strat == 'EX4' and (c5 or c6):
                do_exit = True
            elif exit_strat == 'EX5' and (c5 or c7):
                do_exit = True
            elif exit_strat == 'EX6' and (c6 or c7):
                do_exit = True
            elif exit_strat == 'EX7' and (c5 or c6 or c7):
                do_exit = True
                
            if do_exit:
                current_pos = 0
                exit_price = next_row['Open']
                profit = (exit_price - entry_price) * volume
                cumulative_profit += profit
                trades.append({
                    "type": "LONG", "entry_time": entry_time, "exit_time": next_idx,
                    "entry_price": entry_price, "exit_price": exit_price, "profit": profit
                })
                sig_to_set = 2
                profit_to_set = profit
                
        elif current_pos == -1:
            # --- 売りポジション決済判定 ---
            do_exit = False
            
            c8 = is_lagging_bull                 # 遅行スパン陽転
            c9 = row['Close_gt_minus1']          # 終値が-1σを越える
            c10 = row['Close_gt_21MA']           # 終値が21MA(センターライン)を越える
            
            if exit_strat == 'EX1' and c8:
                do_exit = True
            elif exit_strat == 'EX2' and c9:
                do_exit = True
            elif exit_strat == 'EX3' and c10:
                do_exit = True
            elif exit_strat == 'EX4' and (c8 or c9):
                do_exit = True
            elif exit_strat == 'EX5' and (c8 or c10):
                do_exit = True
            elif exit_strat == 'EX6' and (c9 or c10):
                do_exit = True
            elif exit_strat == 'EX7' and (c8 or c9 or c10):
                do_exit = True
                
            if do_exit:
                current_pos = 0
                exit_price = next_row['Open']
                profit = (entry_price - exit_price) * volume
                cumulative_profit += profit
                trades.append({
                    "type": "SHORT", "entry_time": entry_time, "exit_time": next_idx,
                    "entry_price": entry_price, "exit_price": exit_price, "profit": profit
                })
                sig_to_set = -2
                profit_to_set = profit
                
        df.iat[i + 1, col_sig] = sig_to_set
        df.iat[i + 1, col_prof] = profit_to_set
        df.iat[i + 1, col_pos] = current_pos
        df.iat[i + 1, col_cum] = cumulative_profit
        
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
        current_unrealized_profit = (latest_close - entry_price) * volume * current_pos
        
    # 互換用の集計値
    avg_pnl = total_profit / total_trades if total_trades > 0 else 0.0
    win_trades = wins
    lose_trades = total_trades - wins
    
    # 互換用の残高履歴 (DataFrame)
    balance_history = pd.DataFrame({
        'datetime': df['datetime'] if 'datetime' in df.columns else df.index,
        'balance': equity
    })
    
    # 一括検証スクリプト互換用の metrics 辞書を追加
    metrics = {
        "total_trades": total_trades,
        "win_rate": win_rate / 100.0, # 検証スクリプト側は割合（0.0〜1.0）を想定しているため100で割る
        "total_pnl": total_profit,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd_amount,
        "recovery_factor": total_profit / max_dd_amount if max_dd_amount > 0 else 0.0,
        "avg_pnl": avg_pnl,
        "win_trades": win_trades,
        "lose_trades": lose_trades
    }
    
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
        "trades": trades,
        "metrics": metrics,
        "balance_history": balance_history
    }
