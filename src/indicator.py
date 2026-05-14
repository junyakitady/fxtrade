import pandas as pd

def calculate_super_bollinger(df: pd.DataFrame, period: int = 21) -> pd.DataFrame:
    """
    価格データフレームにスーパーボリンジャーの指標を追加する。
    
    追加される列:
    - center_line: 21期間単純移動平均線
    - plus_1sigma, minus_1sigma 〜 plus_3sigma, minus_3sigma: ボリンジャーバンド
    - chikou_span: 描画用遅行スパン (現在の終値を過去へシフト)
    - past_high_21, past_low_21: シグナル判定用補助列 (過去の高値・安値を現在へシフト)
    """
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # センターライン (21SMA)
    df['center_line'] = df['Close'].rolling(window=period).mean()
    
    # 標準偏差 (ボリンジャーバンド標準の母標準偏差 ddof=0)
    std = df['Close'].rolling(window=period).std(ddof=0)
    
    # バンド計算
    for sigma in [1, 2, 3]:
        df[f'plus_{sigma}sigma'] = df['center_line'] + (std * sigma)
        df[f'minus_{sigma}sigma'] = df['center_line'] - (std * sigma)
        
    # 描画用の遅行スパン (当日の終値を21期間過去へ表示)
    df['chikou_span'] = df['Close'].shift(-period)
    
    # シグナル判定用の補助列 (21期間前のローソク足を当日の行へシフト)
    # 当日の Close > past_high_21 で遅行スパンの上抜けを検知できる
    df['past_high_21'] = df['High'].shift(period)
    df['past_low_21'] = df['Low'].shift(period)
    
    return df

if __name__ == "__main__":
    # 動作確認用スクリプト
    from data_fetcher import fetch_data
    data = fetch_data(period_1h="10d")
    df_1h = data["1h"]
    
    if not df_1h.empty:
        df_ind = calculate_super_bollinger(df_1h)
        print("--- Indicator Added (last 3 rows) ---")
        cols = ['Close', 'center_line', 'plus_1sigma', 'chikou_span', 'past_high_21']
        print(df_ind[cols].tail(3))
