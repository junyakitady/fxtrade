import sys
import os
import pandas as pd
import numpy as np

# プロジェクトのsrcディレクトリをモジュールパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import fetch_data
from indicator import calculate_super_bollinger
from backtest import run_backtest

def generate_dummy_data(periods=1500, freq="1h"):
    """ネットワークエラー（サンドボックス制限）時にリアルな為替ダミーデータを生成する"""
    np.random.seed(42)
    base = 155.0
    returns = np.random.normal(0, 0.0015, periods)
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0.0015, 0.0005, periods)))
    low = close * (1 - np.abs(np.random.normal(0.0015, 0.0005, periods)))
    open_p = np.roll(close, 1)
    open_p[0] = close[0]
    
    df = pd.DataFrame({"Open": open_p, "High": high, "Low": low, "Close": close})
    df.index = pd.date_range(end=pd.Timestamp.now().floor("1h"), periods=periods, freq=freq)
    return df

def get_test_data() -> dict[str, pd.DataFrame]:
    """オンライン取得を試み、失敗時はダミーデータへフォールバックする"""
    data = fetch_data(period_1h="730d", period_1d="5y")
    
    # すべて空の場合（ネットワーク遮断時）はダミーデータを生成
    if (data["1h"] is None or data["1h"].empty) and (data["1d"] is None or data["1d"].empty):
        print("※ネットワーク制限を検知しました。シミュレーション用為替ダミーデータを使用します。\n")
        df_1h = generate_dummy_data(periods=2000, freq="1h")
        df_1d = generate_dummy_data(periods=1200, freq="1D")
        
        # 4時間足の生成
        df_4h = df_1h.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
        
        return {"1h": df_1h, "4h": df_4h, "1d": df_1d}
        
    return data

def run_cross_matrix():
    data = get_test_data()
    
    timeframes = [("1h", "1時間足"), ("4h", "4時間足"), ("1d", "日足")]
    # 本番バックテストエンジンの timeframe 引数を渡すことでそれぞれのモデルを強制適用できる
    models = [
        ("1h", "モデル1 (+2σエクスパンション厳選)"),
        ("4h", "モデル2 (1σウォーク＆スピード撤退)"),
        ("1d", "モデル3 (特大ホームラン特化)")
    ]
    
    print("==========================================================================================")
    print("               3時間軸 × 3シミュレーションモデル 全9パターン クロス検証レポート               ")
    print("==========================================================================================\n")
    
    for tf_code, tf_name in timeframes:
        df = data.get(tf_code)
        if df is None or df.empty:
            continue
            
        df_ind = calculate_super_bollinger(df)
        print(f"■ 時間軸: {tf_name} (ローソク足: {len(df)} 本)")
        print("-" * 90)
        print(f"{'適用シミュレーションモデル':<32} | {'総取引回数 (買/売)':<20} | {'勝率':<10} | {'損益合計 (円)':<15}")
        print("-" * 90)
        
        for model_code, model_name in models:
            res = run_backtest(df_ind, timeframe=model_code, volume=10000)
            trades = res['total_trades']
            l_cnt = res['long_trades']
            s_cnt = res['short_trades']
            rate = res['win_rate']
            prof = res['total_profit']
            
            trade_str = f"{trades} 回 ({l_cnt}/{s_cnt})"
            rate_str = f"{rate:.1f} %"
            prof_str = f"{prof:,.0f} 円"
            
            print(f"{model_name:<30} | {trade_str:<20} | {rate_str:<10} | {prof_str:<15}")
            
        print("-" * 90 + "\n")

if __name__ == "__main__":
    run_cross_matrix()
