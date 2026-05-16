import sys
import os
import pandas as pd
import numpy as np

# プロジェクトのsrcディレクトリをモジュールパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import fetch_data
from indicator import calculate_super_bollinger
from backtest import run_backtest

def get_test_data() -> dict[str, pd.DataFrame]:
    """オンライン取得を厳格に試みる（エラーはそのまま伝播させる）"""
    data = fetch_data(period_1h="730d", period_1d="5y", raise_errors=True)
    return data

def run_cross_matrix():
    try:
        data = get_test_data()
    except Exception as e:
        print("==========================================================================================")
        print("【エラー】バックテスト用データの取得に失敗しました。")
        print(f"原因: {str(e)}")
        print("==========================================================================================")
        sys.exit(1)
    
    timeframes = [("1h", "1時間足"), ("4h", "4時間足"), ("1d", "日足")]
    models = [
        ("1h", "モデル1 (+2σエクスパンション厳選)"),
        ("4h", "モデル2 (1σウォーク＆スピード撤退)"),
        ("1d", "モデル3 (1σセンター撤退モデル)")
    ]
    
    print("======================================================================================================================================================")
    print("                                   3時間軸 × 3シミュレーションモデル 全9パターン クロス検証レポート                                   ")
    print("======================================================================================================================================================\n")
    
    for tf_code, tf_name in timeframes:
        df = data.get(tf_code)
        if df is None or df.empty:
            continue
            
        df_ind = calculate_super_bollinger(df)
        print(f"■ 時間軸: {tf_name} (ローソク足: {len(df)} 本)")
        print("-" * 150)
        print(f"{'適用シミュレーションモデル':<32} | {'総取引回数 (買/売)':<20} | {'勝率':<10} | {'損益合計 (円)':<15} | {'PF':<8} | {'最大ドローダウン (額 / %)'}")
        print("-" * 150)
        
        for model_code, model_name in models:
            res = run_backtest(df_ind, timeframe=model_code, volume=10000)
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
            prof_str = f"{prof:,.0f} 円"
            pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
            dd_str = f"{dd_amt:,.0f} 円 ({dd_pct:.2f}%)"
            
            print(f"{model_name:<30} | {trade_str:<20} | {rate_str:<10} | {prof_str:<15} | {pf_str:<8} | {dd_str}")
            
        print("-" * 150 + "\n")

if __name__ == "__main__":
    run_cross_matrix()
