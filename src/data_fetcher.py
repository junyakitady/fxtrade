import numpy as np
import pandas as pd
import yfinance as yf

def fetch_data(symbol: str = "JPY=X", period_1h: str = "2y", period_1d: str = "5y", raise_errors: bool = False) -> dict[str, pd.DataFrame]:
    """
    Yahoo Financeから価格データを取得し、
    1時間足、4時間足、日足のデータフレームを生成して返す。
    yfinanceによるHTTP通信時は30秒のタイムアウトを指定しハングを防ぐ。
    
    為替以外の個別株等の場合は、日足（1d）データのみを取得してクリーンアップします。
    """
    # 1. 1時間足のダウンロード (為替のみ対象)
    df_1h = None
    if symbol == "JPY=X":
        try:
            df_1h = yf.download(tickers=symbol, period=period_1h, interval="1h", timeout=30)
            if isinstance(df_1h.columns, pd.MultiIndex):
                df_1h.columns = df_1h.columns.get_level_values(0)
            if raise_errors and (df_1h is None or df_1h.empty):
                raise ValueError(f"1時間足データが空です。シンボル '{symbol}' の有効性、またはネットワーク接続を確認してください。")
        except Exception as e:
            if raise_errors:
                raise RuntimeError(f"1時間足データ（期間:{period_1h}）の取得中に例外が発生しました: {str(e)}") from e
            df_1h = None
            
    # 2. 日足のダウンロード (為替・個別株ともに必要)
    try:
        df_1d = yf.download(tickers=symbol, period=period_1d, interval="1d", timeout=30)
        if isinstance(df_1d.columns, pd.MultiIndex):
            df_1d.columns = df_1d.columns.get_level_values(0)
        if raise_errors and (df_1d is None or df_1d.empty):
            raise ValueError(f"日足データが空です。シンボル '{symbol}' の有効性、またはネットワーク接続を確認してください。")
    except Exception as e:
        if raise_errors:
            raise RuntimeError(f"日足データ（期間:{period_1d}）の取得中に例外が発生しました: {str(e)}") from e
        df_1d = None
        
    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close'])
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        df = df.dropna()
        return df

    df_1h_clean = clean_df(df_1h)
    df_1d_clean = clean_df(df_1d)
    
    # 為替（USD/JPY）の場合のみ特殊な前処理および4時間足リサンプルを行う
    if symbol == "JPY=X":
        # 日足特有の「実体が潰れる問題」を解決するため、当日のCloseを翌日のOpenで補正するスマートなハック
        if not df_1d_clean.empty and len(df_1d_clean) > 1:
            close_arr = np.roll(df_1d_clean["Open"].to_numpy(), -1)
            close_arr[-1] = df_1d_clean["Close"].iloc[-1]
            df_1d_clean["Close"] = close_arr
            
            # 補正後のCloseがヒゲを突き抜けて描画矛盾を起こさないようにHigh/Lowも再調整
            df_1d_clean["High"] = np.maximum(df_1d_clean["High"], df_1d_clean["Close"])
            df_1d_clean["Low"] = np.minimum(df_1d_clean["Low"], df_1d_clean["Close"])
            
        # 4時間足の生成 (1時間足をリサンプリング)
        if not df_1h_clean.empty:
            df_4h = df_1h_clean.resample('4h', closed='left', label='left', origin='start_day').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            })
            df_4h_clean = clean_df(df_4h)
        else:
            df_4h_clean = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close'])
    else:
        # 為替以外は1h, 4hは空にする
        df_4h_clean = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close'])
        
    return {
        "1h": df_1h_clean,
        "4h": df_4h_clean,
        "1d": df_1d_clean
    }

if __name__ == "__main__":
    # 為替データテスト
    print("Testing JPY=X data fetch...")
    data_jpy = fetch_data(symbol="JPY=X", period_1h="10d", period_1d="1mo")
    for tf, df in data_jpy.items():
        print(f"--- JPY=X {tf} ---")
        print(f"Shape: {df.shape}")
        if not df.empty:
            print(df.tail(2))
            
    # 個別株データテスト
    print("\nTesting GOOG data fetch...")
    data_goog = fetch_data(symbol="GOOG", period_1d="1mo")
    print(f"--- GOOG 1d ---")
    print(f"Shape: {data_goog['1d'].shape}")
    if not data_goog['1d'].empty:
        print(data_goog['1d'].tail(2))
