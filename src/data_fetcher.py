import numpy as np
import pandas as pd
import yfinance as yf

def fetch_data(symbol: str = "JPY=X", period_1h: str = "730d", period_1d: str = "5y", raise_errors: bool = False) -> dict[str, pd.DataFrame]:
    """
    Yahoo FinanceからUSD/JPYの価格データを取得し、
    1時間足、4時間足、日足のデータフレームを生成して返す。
    休場等の欠損データは削除され、ギャップ詰め処理が行われる。
    
    raise_errors が True の場合、データ取得失敗や空データ受信時に例外を投げます。
    False の場合は安全に空のデータフレームへフォールバックします。
    """
    ticker = yf.Ticker(symbol)
    
    try:
        df_1h = ticker.history(period=period_1h, interval="1h")
        if raise_errors and (df_1h is None or df_1h.empty):
            raise ValueError(f"1時間足データが空です。シンボル '{symbol}' の有効性、またはネットワーク接続を確認してください。")
    except Exception as e:
        if raise_errors:
            raise RuntimeError(f"1時間足データ（期間:{period_1h}）の取得中に例外が発生しました: {str(e)}") from e
        df_1h = None
        
    try:
        df_1d = ticker.history(period=period_1d, interval="1d")
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
        
    return {
        "1h": df_1h_clean,
        "4h": df_4h_clean,
        "1d": df_1d_clean
    }

if __name__ == "__main__":
    data = fetch_data(period_1h="10d", period_1d="1mo")
    for tf, df in data.items():
        print(f"--- {tf} ---")
        print(f"Shape: {df.shape}")
        if not df.empty:
            print(df.tail(2))
