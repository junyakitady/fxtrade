import pandas as pd
import yfinance as yf

def fetch_data(symbol: str = "JPY=X", period_1h: str = "60d", period_1d: str = "2y") -> dict[str, pd.DataFrame]:
    """
    Yahoo FinanceからUSD/JPYの価格データを取得し、
    1時間足、4時間足、日足のデータフレームを生成して返す。
    休場等の欠損データは削除され、ギャップ詰め処理が行われる。
    """
    ticker = yf.Ticker(symbol)
    
    # 1時間足の取得 (過去60日分など)
    df_1h = ticker.history(period=period_1h, interval="1h")
    
    # 日足の取得 (過去2年分など、十分な期間)
    df_1d = ticker.history(period=period_1d, interval="1d")
    
    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close'])
        # 必要な列のみ抽出
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        # 欠損値の削除（休場のギャップ詰め）
        df = df.dropna()
        return df

    df_1h_clean = clean_df(df_1h)
    df_1d_clean = clean_df(df_1d)
    
    # 4時間足の生成 (1時間足をリサンプリング)
    if not df_1h_clean.empty:
        # 4時間ごとの集約。データがない期間（週末等）も行ができるため再度clean_dfを通す
        df_4h = df_1h_clean.resample('4h').agg({
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
    # 動作確認用スクリプト
    data = fetch_data()
    for tf, df in data.items():
        print(f"--- {tf} ---")
        print(f"Shape: {df.shape}")
        if not df.empty:
            print(df.tail(2))
