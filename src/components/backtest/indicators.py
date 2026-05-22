import pandas as pd
import numpy as np

def calculate_super_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレームにスーパーボリンジャーの指標と判定用フラグを追加する。
    
    Args:
        df: 前処理済みのデータフレーム (Columns: datetime, Open, High, Low, Close)
        
    Returns:
        pd.DataFrame: 指標が追加されたデータフレーム
    """
    df = df.copy()
    
    # 21MA (中心線)
    df['21MA'] = df['Close'].rolling(window=21).mean()
    
    # 標準偏差 (21期間)
    df['StdDev'] = df['Close'].rolling(window=21).std()
    
    # ボリンジャーバンド
    df['+1σ'] = df['21MA'] + 1 * df['StdDev']
    df['+2σ'] = df['21MA'] + 2 * df['StdDev']
    df['-1σ'] = df['21MA'] - 1 * df['StdDev']
    df['-2σ'] = df['21MA'] - 2 * df['StdDev']
    
    # 遅行スパン比較用 (21期間前の終値/高値/安値)
    df['Close_21_ago'] = df['Close'].shift(21)
    df['High_21_ago'] = df['High'].shift(21)
    df['Low_21_ago'] = df['Low'].shift(21)
    
    # 遅行スパン陽転・陰転状態 (デフォルト: 終値ベース)
    df['Lagging_Bullish'] = df['Close'] > df['Close_21_ago']
    df['Lagging_Bearish'] = df['Close'] < df['Close_21_ago']
    
    # バンド幅変化 (エクスパンション判定用)
    # +2σが前値より大きく、-2σが前値より小さい（外側に広がっている）
    df['+2σ_diff'] = df['+2σ'].diff()
    df['-2σ_diff'] = df['-2σ'].diff()
    df['Expansion'] = (df['+2σ_diff'] > 0) & (df['-2σ_diff'] < 0)
    
    # 終値とバンドの比較フラグ
    df['Close_gt_plus1'] = df['Close'] > df['+1σ']
    df['Close_gt_plus2'] = df['Close'] > df['+2σ']
    df['Close_lt_minus1'] = df['Close'] < df['-1σ']
    df['Close_lt_minus2'] = df['Close'] < df['-2σ']
    
    # エグジット判定用フラグ
    df['Close_lt_plus1'] = df['Close'] < df['+1σ']
    df['Close_lt_21MA'] = df['Close'] < df['21MA']
    
    df['Close_gt_minus1'] = df['Close'] > df['-1σ']
    df['Close_gt_21MA'] = df['Close'] > df['21MA']
    
    return df

if __name__ == "__main__":
    # 簡易テスト
    from data_loader import get_usd_jpy_data
    try:
        df = get_usd_jpy_data('1d', '5y')
        df_indicators = calculate_super_bollinger(df)
        print("Indicators calculated successfully.")
        print(df_indicators[['datetime', 'Close', '21MA', '+2σ', '-2σ', 'Lagging_Bullish', 'Expansion']].tail(10))
    except Exception as e:
        print(f"Error: {e}")
