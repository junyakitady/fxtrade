import os
import pandas as pd
import yfinance as yf
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def get_usd_jpy_data(interval: str, period: str = '5y', force_download: bool = False) -> pd.DataFrame:
    """
    USD/JPYのヒストリカルデータを取得し、前処理（休場ギャップ排除、インデックス連番化）を行う。
    
    Args:
        interval: '1h', '4h', '1d'
        period: データの取得期間 ('2y', '5y' など)
        force_download: Trueの場合、キャッシュを無視して再ダウンロードする
        
    Returns:
        pd.DataFrame: 前処理済みのデータフレーム (Columns: datetime, Open, High, Low, Close, Volume)
                      Indexは連番(0, 1, 2...)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1hと4hはyfinanceの制限により最大2年
    if interval in ['1h', '4h'] and period not in ['1y', '2y']:
        # 2年以上が指定された場合は自動的に2年に制限
        period = '2y'
        
    cache_file = os.path.join(DATA_DIR, f"USDJPY_{interval}_{period}.csv")
    
    if not force_download and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
        
    print(f"Downloading USD/JPY data for interval={interval}, period={period}...")
    ticker = "JPY=X" # yfinanceでのUSD/JPY
    
    if interval == '4h':
        # 4hは1hからリサンプルして作成する
        raw_df = yf.download(tickers=ticker, period=period, interval='1h')
        if raw_df.empty:
            raise ValueError("Failed to download 1h data for 4h resampling.")
        
        # マルチインデックス対策（yfinanceの仕様変更対応）
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
            
        df = resample_to_4h(raw_df)
    else:
        raw_df = yf.download(tickers=ticker, period=period, interval=interval)
        if raw_df.empty:
            raise ValueError(f"Failed to download data for interval={interval}.")
            
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        df = raw_df.copy()
        
    # 前処理
    df = preprocess_data(df)
    
    # キャッシュ保存
    df.to_csv(cache_file, index=False)
    print(f"Saved data to {cache_file}")
    
    return df

def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    1時間足データを4時間足にリサンプルする。
    基準時間は 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 (UTC想定)
    """
    # インデックスがDatetimeIndexであることを確認
    if not isinstance(df_1h.index, pd.DatetimeIndex):
        df_1h.index = pd.to_datetime(df_1h.index)
        
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    # 4時間足にリサンプル (閉区間は左、ラベルも左)
    # origin='start_day' で一日の始まり(00:00)を起点にする
    df_4h = df_1h.resample('4h', closed='left', label='left', origin='start_day').agg(ohlc_dict)
    
    # データがない時間は除外
    df_4h = df_4h.dropna(subset=['Open', 'High', 'Low', 'Close'])
    return df_4h

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    休場ギャップの排除とインデックスの再設定を行う。
    """
    # インデックスがDatetimeIndexの場合、カラムに退避
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df = df.rename(columns={'Date': 'datetime', 'Datetime': 'datetime'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'datetime'})
    elif 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'datetime'})
        
    # カラム名を標準化
    df.columns = [col.capitalize() if col in ['open', 'high', 'low', 'close', 'volume'] else col for col in df.columns]
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'datetime'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'datetime'})
        
    # 必要なカラムのみ抽出
    required_cols = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    # 存在しないカラム（Volumeなどがない場合がある）への対応
    actual_cols = [col for col in required_cols if col in df.columns]
    df = df[actual_cols]
    
    # 欠損値の削除（休場ギャップ等の排除）
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    # 重複の削除
    df = df.drop_duplicates(subset=['datetime'])
    
    # 日時順にソート
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # インデックスは連番（0, 1, 2...）になっている。
    # これにより、インデックスの差がそのままバーの数になる。
    return df

if __name__ == "__main__":
    # 簡易テスト
    try:
        print("Testing daily data fetch...")
        df_d = get_usd_jpy_data('1d', '5y', force_download=True)
        print(f"Daily data shape: {df_d.shape}")
        print(df_d.head())
        
        print("\nTesting 1h data fetch...")
        df_1h = get_usd_jpy_data('1h', '2y', force_download=True)
        print(f"1h data shape: {df_1h.shape}")
        print(df_1h.head())
        
        print("\nTesting 4h data fetch (resampled from 1h)...")
        df_4h = get_usd_jpy_data('4h', '2y', force_download=True)
        print(f"4h data shape: {df_4h.shape}")
        print(df_4h.head())
        
    except Exception as e:
        print(f"Error during testing: {e}")
