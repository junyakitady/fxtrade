import os
import pandas as pd
import sys

# data_fetcherがインポートできるように親パスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_fetcher import fetch_data

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def get_usd_jpy_data(interval: str, period: str = '5y', force_download: bool = False) -> pd.DataFrame:
    """
    USD/JPYのヒストリカルデータを取得し、キャッシュする。
    データ取得とリサンプル、日足Close補正のプロセスは本番モジュール(data_fetcher)へ完全一元化。
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1hと4hはyfinanceの制限により最大2年
    if interval in ['1h', '4h'] and period not in ['1y', '2y']:
        period = '2y'
        
    cache_file = os.path.join(DATA_DIR, f"USDJPY_{interval}_{period}.csv")
    
    if not force_download and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
        
    print(f"Downloading USD/JPY data for interval={interval}, period={period}...")
    
    # data_fetcherのfetch_dataを使って本番と全く同じプロセスで取得
    if interval == '1d':
        data = fetch_data(period_1d=period, raise_errors=True)
        df = data['1d']
    elif interval == '4h':
        data = fetch_data(period_1h=period, raise_errors=True)
        df = data['4h']
    else:
        data = fetch_data(period_1h=period, raise_errors=True)
        df = data['1h']
        
    # インデックスを「datetime」列に変更して検証ツール互換にする
    df = df.reset_index().rename(columns={'Date': 'datetime', 'Datetime': 'datetime'})
    
    # 重複と欠損値の除去（検証用のクリーン化）
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df = df.drop_duplicates(subset=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # キャッシュ保存
    df.to_csv(cache_file, index=False)
    print(f"Saved data to {cache_file}")
    
    return df

if __name__ == "__main__":
    # 簡易テスト
    try:
        df_1h = get_usd_jpy_data('1h', '2y', force_download=True)
        print(f"1h data shape: {df_1h.shape}")
        print(df_1h.head(2))
        
        df_1d = get_usd_jpy_data('1d', '5y', force_download=True)
        print(f"1d data shape: {df_1d.shape}")
        print(df_1d.head(2))
    except Exception as e:
        print(f"Error during testing: {e}")
