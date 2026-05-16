import sys
import os
import itertools
import pandas as pd
import numpy as np

# 親ディレクトリ (src) をモジュールパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import fetch_data
from indicator import calculate_super_bollinger

def run_grid_search(df_ind: pd.DataFrame, volume: int = 10000) -> list[dict]:
    """
    条件1(遅行スパン)を必須(Must)とし、
    「1トレードあたりの平均利益」が高い組み合わせを探索する。
    """
    df = df_ind.copy()
    
    df['m2s_diff'] = df['minus_2sigma'].diff()
    df['p2s_diff'] = df['plus_2sigma'].diff()
    df['cl_diff'] = df['center_line'].diff()
    
    # 各条件の二値配列を事前計算
    # 1: 遅行スパン (必須)
    e1_l = df['Close'] > df['past_high_21']
    e1_s = df['Close'] < df['past_low_21']
    # 2: 1σ
    e2_l = df['Close'] > df['plus_1sigma']
    e2_s = df['Close'] < df['minus_1sigma']
    # 3: 2σ
    e3_l = df['Close'] > df['plus_2sigma']
    e3_s = df['Close'] < df['minus_2sigma']
    # 4: 本物のエクスパンション (上下に広がる)
    e4 = (df['m2s_diff'] < 0) & (df['p2s_diff'] > 0)
    e4_l = e4
    e4_s = e4
    # 5: センターライン傾き
    e5_l = df['cl_diff'] > 0
    e5_s = df['cl_diff'] < 0
    
    # 決済条件
    x6_l = df['Close'] < df['past_low_21']
    x6_s = df['Close'] > df['past_high_21']
    x7_l = df['Close'] < df['plus_1sigma']
    x7_s = df['Close'] > df['minus_1sigma']
    x8_l = df['Close'] < df['center_line']
    x8_s = df['Close'] > df['center_line']
    
    N = len(df)
    E_L = np.column_stack([e1_l, e2_l, e3_l, e4_l, e5_l])
    E_S = np.column_stack([e1_s, e2_s, e3_s, e4_s, e5_s])
    
    X_L = np.column_stack([x6_l, x7_l, x8_l])
    X_S = np.column_stack([x6_s, x7_s, x8_s])
    
    close_arr = df['Close'].to_numpy()
    valid_mask = (~df['plus_1sigma'].isna() & ~df['past_high_21'].isna() & ~df['cl_diff'].isna()).to_numpy()
    
    # エントリーの組み合わせ (条件1=インデックス0 は必ず含める)
    entry_combos = []
    for r in range(5):
        for c in itertools.combinations([1, 2, 3, 4], r):
            entry_combos.append((0,) + c)
            
    # 決済の組み合わせ (0〜2の中から1つ以上選ぶ)
    exit_combos = []
    for r in range(1, 4):
        for c in itertools.combinations(range(3), r):
            exit_combos.append(c)
            
    results = []
    
    for e_indices in entry_combos:
        for x_indices in exit_combos:
            for exit_mode in ['OR', 'AND']:
                if len(x_indices) == 1 and exit_mode == 'AND':
                    continue
                    
                current_pos = 0
                entry_price = 0.0
                profit_sum = 0.0
                trade_count = 0
                win_count = 0
                
                for i in range(1, N):
                    if not valid_mask[i]:
                        continue
                        
                    cp = close_arr[i]
                    
                    if current_pos == 0:
                        if np.all(E_L[i, e_indices]):
                            current_pos = 1
                            entry_price = cp
                        elif np.all(E_S[i, e_indices]):
                            current_pos = -1
                            entry_price = cp
                    elif current_pos == 1:
                        do_exit = np.any(X_L[i, x_indices]) if exit_mode == 'OR' else np.all(X_L[i, x_indices])
                        if do_exit:
                            current_pos = 0
                            prof = (cp - entry_price) * volume
                            profit_sum += prof
                            trade_count += 1
                            if prof > 0: win_count += 1
                    elif current_pos == -1:
                        do_exit = np.any(X_S[i, x_indices]) if exit_mode == 'OR' else np.all(X_S[i, x_indices])
                        if do_exit:
                            current_pos = 0
                            prof = (entry_price - cp) * volume
                            profit_sum += prof
                            trade_count += 1
                            if prof > 0: win_count += 1
                            
                win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0.0
                avg_profit = (profit_sum / trade_count) if trade_count > 0 else 0.0
                
                ENTRY_LABELS = ["1(遅行スパン)", "2(+1σ)", "3(+2σ)", "4(バンド拡大)", "5(中心線傾き)"]
                EXIT_LABELS = ["6(遅行逆転)", "7(1σ逆値)", "8(中心線割れ)"]
                
                e_names = [ENTRY_LABELS[idx] for idx in e_indices]
                x_names = [EXIT_LABELS[idx] for idx in x_indices]
                e_str = " AND ".join(e_names)
                x_str = f" {exit_mode} ".join(x_names) if len(x_names) > 1 else x_names[0]
                
                # 極端な少回数ノイズを除外するため、最低取引回数のフィルタを設けるか純粋に記録
                results.append({
                    "entry_str": e_str,
                    "exit_str": x_str,
                    "trades": trade_count,
                    "win_rate": win_rate,
                    "profit": profit_sum,
                    "avg_profit": avg_profit
                })
                
    # 1トレードあたりの平均利益 (avg_profit) でソート
    results.sort(key=lambda x: x['avg_profit'], reverse=True)
    return results[:5]

if __name__ == "__main__":
    data = fetch_data(period_1h="730d", period_1d="5y")
    for tf, df in data.items():
        if not df.empty:
            df_ind = calculate_super_bollinger(df)
            print(f"\n================ {tf} 平均利益トップ5 (条件1Must) ================")
            top_results = run_grid_search(df_ind)
            for rank, res in enumerate(top_results, 1):
                print(f"Rank {rank}: エントリー=[{res['entry_str']}] ｜ 決済=[{res['exit_str']}]")
                print(f"         平均利益={res['avg_profit']:,.1f}円 ｜ 取引={res['trades']}回 ｜ 勝率={res['win_rate']:.1f}% ｜ 損益={res['profit']:,.0f}円")
