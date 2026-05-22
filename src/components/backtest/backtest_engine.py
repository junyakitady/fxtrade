import pandas as pd
import numpy as np

def run_backtest(df: pd.DataFrame, entry_strat: str, exit_strat: str, lot_size: int = 10000, initial_capital: float = 1000000.0, lagging_type: str = 'close') -> dict:
    """
    指定されたエントリー戦略とエグジット戦略でバックテストを実行する。
    
    Args:
        df: インジケータ計算済みのデータフレーム
        entry_strat: 'E1', 'E2', 'E3', 'E4'
        exit_strat: 'EX1' ~ 'EX7'
        lot_size: 取引通貨単位 (デフォルト 10,000)
        initial_capital: 初期資金 (デフォルト 1,000,000円)
        
    Returns:
        dict: バックテスト結果の指標と、トレード履歴
    """
    df = df.copy()
    
    # ポジション状態
    # None (ノーポジ), 'Long' (ロング保有), 'Short' (ショート保有)
    position = None
    entry_price = 0.0
    entry_time = None
    
    trades = []
    current_balance = initial_capital
    balances = [current_balance]
    balance_times = [df.loc[0, 'datetime'] if len(df) > 0 else None]
    
    # 21期間のデータが必要なため、インデックス21から開始 (遅行スパンは過去21期間必要なので、実質インデックス21から有効)
    # yfinanceのデータにはNaNが含まれる可能性があるため、インジケータが正常に計算できている最初の行を探す
    start_idx = 0
    for i in range(len(df)):
        if not pd.isna(df.loc[i, '21MA']) and not pd.isna(df.loc[i, 'Close_21_ago']):
            # 前日比（エクスパンション用のdiff）も必要なので、そこから+1する
            start_idx = i + 1
            break
            
    if start_idx >= len(df):
        return {"metrics": None, "trades": []}
        
    for t in range(start_idx, len(df) - 1):
        # 現在の行
        row = df.iloc[t]
        # 次の行 (エントリー/エグジットの約定用)
        next_row = df.iloc[t + 1]
        
        # 遅行スパンの判定定義の切り替え
        if lagging_type == 'close':
            is_lagging_bull = row['Close'] > row['Close_21_ago']
            is_lagging_bear = row['Close'] < row['Close_21_ago']
        elif lagging_type == 'high_low':
            is_lagging_bull = row['Close'] > row['High_21_ago']
            is_lagging_bear = row['Close'] < row['Low_21_ago']
            
        # 1. エントリー判定 (ポジションなしの場合)
        if position is None:
            # --- ロングエントリー条件判定 ---
            is_long_entry = False
            # 必須条件: 遅行スパン陽転状態 かつ エクスパンション状態
            if is_lagging_bull and row['Expansion']:
                if entry_strat == 'E1':
                    is_long_entry = True
                elif entry_strat == 'E2' and row['Close_gt_plus1']:
                    is_long_entry = True
                elif entry_strat == 'E3' and row['Close_gt_plus2']:
                    is_long_entry = True
                elif entry_strat == 'E4' and row['Close_gt_plus1'] and row['Close_gt_plus2']:
                    is_long_entry = True
                    
            # --- ショートエントリー条件判定 ---
            is_short_entry = False
            # 必須条件: 遅行スパン陰転状態 かつ エクスパンション状態
            if is_lagging_bear and row['Expansion']:
                if entry_strat == 'E1':
                    is_short_entry = True
                elif entry_strat == 'E2' and row['Close_lt_minus1']:
                    is_short_entry = True
                elif entry_strat == 'E3' and row['Close_lt_minus2']:
                    is_short_entry = True
                elif entry_strat == 'E4' and row['Close_lt_minus1'] and row['Close_lt_minus2']:
                    is_short_entry = True
                    
            # エントリー実行 (次の足の始値で約定)
            if is_long_entry:
                position = 'Long'
                entry_price = next_row['Open']
                entry_time = next_row['datetime']
            elif is_short_entry:
                position = 'Short'
                entry_price = next_row['Open']
                entry_time = next_row['datetime']
                
        # 2. エグジット判定 (ポジション保有中の場合)
        elif position == 'Long':
            is_exit = False
            
            # 各条件のフラグ
            c5 = is_lagging_bear          # 遅行スパン陰転
            c6 = row['Close_lt_plus1']    # 終値が+1σを割れる
            c7 = row['Close_lt_21MA']     # 終値が21MA中心線を割れる
            
            if exit_strat == 'EX1' and c5:
                is_exit = True
            elif exit_strat == 'EX2' and c6:
                is_exit = True
            elif exit_strat == 'EX3' and c7:
                is_exit = True
            elif exit_strat == 'EX4' and (c5 or c6):
                is_exit = True
            elif exit_strat == 'EX5' and (c5 or c7):
                is_exit = True
            elif exit_strat == 'EX6' and (c6 or c7):
                is_exit = True
            elif exit_strat == 'EX7' and (c5 or c6 or c7):
                is_exit = True
                
            # エグジット実行 (次の足の始値で約定)
            if is_exit:
                exit_price = next_row['Open']
                pnl = (exit_price - entry_price) * lot_size
                current_balance += pnl
                
                trades.append({
                    'type': 'Long',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': next_row['datetime'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'balance': current_balance
                })
                balances.append(current_balance)
                balance_times.append(next_row['datetime'])
                position = None
                
        elif position == 'Short':
            is_exit = False
            
            # 各条件のフラグ (ショート用は逆判定)
            c5 = is_lagging_bull          # 遅行スパン陽転
            c6 = row['Close_gt_minus1']   # 終値が-1σを超える
            c7 = row['Close_gt_21MA']     # 終値が21MA中心線を超える
            
            if exit_strat == 'EX1' and c5:
                is_exit = True
            elif exit_strat == 'EX2' and c6:
                is_exit = True
            elif exit_strat == 'EX3' and c7:
                is_exit = True
            elif exit_strat == 'EX4' and (c5 or c6):
                is_exit = True
            elif exit_strat == 'EX5' and (c5 or c7):
                is_exit = True
            elif exit_strat == 'EX6' and (c6 or c7):
                is_exit = True
            elif exit_strat == 'EX7' and (c5 or c6 or c7):
                is_exit = True
                
            # エグジット実行 (次の足の始値で約定)
            if is_exit:
                exit_price = next_row['Open']
                pnl = (entry_price - exit_price) * lot_size
                current_balance += pnl
                
                trades.append({
                    'type': 'Short',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': next_row['datetime'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'balance': current_balance
                })
                balances.append(current_balance)
                balance_times.append(next_row['datetime'])
                position = None
                
    # 最終行でポジションを保有している場合は、最終行の終値で強制クローズ
    if position is not None:
        last_row = df.iloc[-1]
        exit_price = last_row['Close']
        if position == 'Long':
            pnl = (exit_price - entry_price) * lot_size
        else:
            pnl = (entry_price - exit_price) * lot_size
            
        current_balance += pnl
        trades.append({
            'type': position,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': last_row['datetime'],
            'exit_price': exit_price,
            'pnl': pnl,
            'balance': current_balance,
            'forced_close': True
        })
        balances.append(current_balance)
        balance_times.append(last_row['datetime'])
        
    # 指標の算出
    metrics = calculate_metrics(trades, balances, initial_capital)
    
    # 資産推移をデータフレーム化
    balance_df = pd.DataFrame({
        'datetime': balance_times,
        'balance': balances
    })
    
    return {
        'metrics': metrics,
        'trades': trades,
        'balance_history': balance_df
    }

def calculate_metrics(trades: list, balances: list, initial_capital: float) -> dict:
    """
    トレード履歴からパフォーマンス指標を算出する。
    """
    total_trades = len(trades)
    if total_trades == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'win_trades': 0,
            'lose_trades': 0
        }
        
    pnl_list = [t['pnl'] for t in trades]
    total_pnl = sum(pnl_list)
    avg_pnl = total_pnl / total_trades
    
    win_trades = sum(1 for p in pnl_list if p > 0)
    lose_trades = sum(1 for p in pnl_list if p <= 0)
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    
    # プロフィットファクター (PF)
    gross_profit = sum(p for p in pnl_list if p > 0)
    gross_loss = abs(sum(p for p in pnl_list if p < 0))
    
    if gross_loss == 0:
        profit_factor = float('inf') if gross_profit > 0 else 1.0
    else:
        profit_factor = gross_profit / gross_loss
        
    # 最大ドローダウン (金額ベース)
    balances_arr = np.array(balances)
    peaks = np.maximum.accumulate(balances_arr)
    drawdowns = peaks - balances_arr
    max_dd = np.max(drawdowns)
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'win_trades': win_trades,
        'lose_trades': lose_trades
    }

if __name__ == "__main__":
    # 簡易テスト
    from data_loader import get_usd_jpy_data
    from indicators import calculate_super_bollinger
    
    try:
        df = get_usd_jpy_data('1d', '5y')
        df = calculate_super_bollinger(df)
        
        # E2 & EX4 の組み合わせでテスト
        result = run_backtest(df, 'E2', 'EX4')
        metrics = result['metrics']
        print("Backtest metrics for E2 & EX4:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"Total trades: {len(result['trades'])}")
        if len(result['trades']) > 0:
            print("First trade:", result['trades'][0])
            print("Last trade:", result['trades'][-1])
    except Exception as e:
        print(f"Error: {e}")
