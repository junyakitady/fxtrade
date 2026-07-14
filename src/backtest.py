import pandas as pd
import numpy as np

def _generate_backtest_signals(df: pd.DataFrame, entry_strat: str, exit_strat: str, lagging_type: str) -> pd.DataFrame:
    """
    バックテスト用の取引判定シグナル（エントリー・エグジット）をベクトル演算で生成する。
    """
    df = df.copy()
    
    # シグナル列の初期化
    df['entry_long'] = False
    df['entry_short'] = False
    df['exit_long'] = False
    df['exit_short'] = False
    
    if entry_strat == "MACD_DOTEN":
        df['entry_long'] = df.get('macd_gc', False)
        df['entry_short'] = df.get('macd_dc', False)
        df['exit_long'] = df.get('macd_dc', False)
        df['exit_short'] = df.get('macd_gc', False)
        return df
        
    # 遅行スパン判定（終値基準 / 高安基準の解決）
    if lagging_type == 'close':
        is_lagging_bull = df['Close'] > df.get('Close_21_ago', df['Close'].shift(21))
        is_lagging_bear = df['Close'] < df.get('Close_21_ago', df['Close'].shift(21))
    else: # 'high_low'
        is_lagging_bull = df['Close'] > df.get('High_21_ago', df['High'].shift(21))
        is_lagging_bear = df['Close'] < df.get('Low_21_ago', df['Low'].shift(21))
        
    # --- 買いエントリー条件 (entry_long) ---
    is_long = pd.Series(False, index=df.index)
    if entry_strat == 'E1':
        is_long = is_lagging_bull
    elif entry_strat == 'E2':
        is_long = is_lagging_bull & df.get('Close_gt_plus1', False)
    elif entry_strat == 'E3':
        is_long = is_lagging_bull & df.get('Close_gt_plus2', False)
    elif entry_strat == 'E4':
        is_long = is_lagging_bull & df.get('Expansion', False)
    elif entry_strat == 'E5':
        is_long = is_lagging_bull & df.get('Expansion', False) & df.get('Close_gt_plus1', False)
    elif entry_strat == 'E6':
        is_long = is_lagging_bull & df.get('Expansion', False) & df.get('Close_gt_plus2', False)
        
    # --- 売りエントリー条件 (entry_short) ---
    is_short = pd.Series(False, index=df.index)
    if entry_strat == 'E1':
        is_short = is_lagging_bear
    elif entry_strat == 'E2':
        is_short = is_lagging_bear & df.get('Close_lt_minus1', False)
    elif entry_strat == 'E3':
        is_short = is_lagging_bear & df.get('Close_lt_minus2', False)
    elif entry_strat == 'E4':
        is_short = is_lagging_bear & df.get('Expansion', False)
    elif entry_strat == 'E5':
        is_short = is_lagging_bear & df.get('Expansion', False) & df.get('Close_lt_minus1', False)
    elif entry_strat == 'E6':
        is_short = is_lagging_bear & df.get('Expansion', False) & df.get('Close_lt_minus2', False)

    # 21MA同方向フィルターの適用 (EX3, EX5, EX6, EX7が対象)
    if exit_strat in ['EX3', 'EX5', 'EX6', 'EX7']:
        is_long = is_long & (df['Close'] > df.get('center_line', df['Close']))
        is_short = is_short & (df['Close'] < df.get('center_line', df['Close']))

    df['entry_long'] = is_long
    df['entry_short'] = is_short

    # --- 買いエグジット条件 (exit_long) ---
    c5 = is_lagging_bear
    c6 = df.get('Close_lt_plus1', False)
    c7 = df.get('Close_lt_21MA', False)
    
    exit_l = pd.Series(False, index=df.index)
    if exit_strat == 'EX1':
        exit_l = c5
    elif exit_strat == 'EX2':
        exit_l = c6
    elif exit_strat == 'EX3':
        exit_l = c7
    elif exit_strat == 'EX4':
        exit_l = c5 | c6
    elif exit_strat == 'EX5':
        exit_l = c5 | c7
    elif exit_strat == 'EX6':
        exit_l = c6 | c7
    elif exit_strat == 'EX7':
        exit_l = c5 | c6 | c7
        
    # --- 売りエグジット条件 (exit_short) ---
    c8 = is_lagging_bull
    c9 = df.get('Close_gt_minus1', False)
    c10 = df.get('Close_gt_21MA', False)
    
    exit_s = pd.Series(False, index=df.index)
    if exit_strat == 'EX1':
        exit_s = c8
    elif exit_strat == 'EX2':
        exit_s = c9
    elif exit_strat == 'EX3':
        exit_s = c10
    elif exit_strat == 'EX4':
        exit_s = c8 | c9
    elif exit_strat == 'EX5':
        exit_s = c8 | c10
    elif exit_strat == 'EX6':
        exit_s = c9 | c10
    elif exit_strat == 'EX7':
        exit_s = c8 | c9 | c10

    df['exit_long'] = exit_l
    df['exit_short'] = exit_s

    return df

def run_backtest(df: pd.DataFrame, entry_strat: str = None, exit_strat: str = None, lagging_type: str = None, timeframe: str = None, volume: int = 10000, initial_capital: float = 1000000.0, lot_size: int = None) -> dict:
    """
    スーパーボリンジャー算出済みのデータフレーム上で売買シミュレーションを実行する。
    本番UIおよび一括検証の共通バックテストシミュレータ。
    
    タイムフレーム、エントリー・決済パラメータ、遅行スパン基準に基づいた汎用シミュレーションを実行。
    翌足始値約定モデル（厳密版）＋ 21MAエグジット時の21MA同方向エントリーフィルター付き。
    """
    if lot_size is not None:
        volume = lot_size

    # timeframeから判断できるように早めの解決、またはカラム確認
    is_macd = (timeframe == "1h_macd_doten" or entry_strat == "MACD_DOTEN")
    required_col = 'macd' if is_macd else 'plus_1sigma'
    
    if df is None or df.empty or required_col not in df.columns:
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
        elif timeframe == "1h_macd_doten":
            entry_strat = "MACD_DOTEN"
            exit_strat = "MACD_DOTEN"
            lagging_type = "none"
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
    
    # 事前にシグナル（entry_long, entry_short, exit_long, exit_short）をベクトル計算で生成
    df_signals = _generate_backtest_signals(df, entry_strat, exit_strat, lagging_type)
    
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
    
    is_macd_doten = (entry_strat == "MACD_DOTEN")
    
    for i in range(1, len(df) - 1):
        row = df_signals.iloc[i]
        idx = df_signals.index[i]
        next_row = df_signals.iloc[i + 1]
        next_idx = df_signals.index[i + 1]
        
        # 必要なカラムの存在チェックおよびNaNチェック
        if is_macd_doten:
            if 'macd' not in df.columns or pd.isna(row.get('macd')) or pd.isna(row.get('macd_gc')):
                df.iat[i + 1, col_pos] = current_pos
                df.iat[i + 1, col_cum] = cumulative_profit
                continue
        else:
            if 'plus_1sigma' not in df.columns or pd.isna(row.get('plus_1sigma')):
                df.iat[i + 1, col_pos] = current_pos
                df.iat[i + 1, col_cum] = cumulative_profit
                continue
                
        sig_to_set = 0
        profit_to_set = 0.0
        
        if is_macd_doten:
            # MACDドテン約定モデル
            if row['entry_long']: # 買いドテンシグナル
                if current_pos == -1: # ショートポジションの決済
                    exit_price = next_row['Open']
                    profit = (entry_price - exit_price) * volume
                    cumulative_profit += profit
                    trades.append({
                        "type": "SHORT", "entry_time": entry_time, "exit_time": next_idx,
                        "entry_price": entry_price, "exit_price": exit_price, "profit": profit
                    })
                    profit_to_set = profit
                current_pos = 1
                entry_price = next_row['Open']
                entry_time = next_idx
                sig_to_set = 1
            elif row['entry_short']: # 売りドテンシグナル
                if current_pos == 1: # ロングポジションの決済
                    exit_price = next_row['Open']
                    profit = (exit_price - entry_price) * volume
                    cumulative_profit += profit
                    trades.append({
                        "type": "LONG", "entry_time": entry_time, "exit_time": next_idx,
                        "entry_price": entry_price, "exit_price": exit_price, "profit": profit
                    })
                    profit_to_set = profit
                current_pos = -1
                entry_price = next_row['Open']
                entry_time = next_idx
                sig_to_set = -1
        else:
            # 通常の決済＆エントリーモデル
            if current_pos == 0:
                # 新規エントリー
                if row['entry_long']:
                    current_pos = 1
                    entry_price = next_row['Open']
                    entry_time = next_idx
                    sig_to_set = 1
                elif row['entry_short']:
                    current_pos = -1
                    entry_price = next_row['Open']
                    entry_time = next_idx
                    sig_to_set = -1
            elif current_pos == 1:
                # 買い決済
                if row['exit_long']:
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
                # 売り決済
                if row['exit_short']:
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
