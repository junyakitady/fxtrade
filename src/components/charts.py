import pandas as pd
import numpy as np
import plotly.graph_objects as go

def create_super_bollinger_chart(df: pd.DataFrame, title: str = "スーパーボリンジャー", initial_display_candles: int = 64) -> go.Figure:
    """
    Plotlyを用いてローソク足チャート、スーパーボリンジャー各指標、売買マーカーを描画する。
    休場ギャップ排除のためX軸をカテゴリ軸として処理する。
    買いおよび売りのエントリー・決済の全4種類のシグナルマーカーを重畳描画する。
    """
    fig = go.Figure()
    
    if df is None or df.empty:
        fig.update_layout(title=f"{title} (データなし)", template="plotly_dark")
        return fig
        
    # 日中足（1時間足、4時間足など）かどうかの自動判別（タイトル依存の解消）
    is_intraday = False
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        median_interval = pd.Series(df.index).diff().median()
        if median_interval < pd.Timedelta(days=1):
            is_intraday = True
            
    if is_intraday:
        x_strings = []
        base_year = df.index[0].year
        for idx in df.index:
            base_str = idx.strftime('%m-%d %H:%M')
            unique_str = base_str + ('\u200b' * (idx.year - base_year))
            x_strings.append(unique_str)
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            x_strings = df.index.strftime('%Y-%m-%d')
        else:
            x_strings = [str(idx) for idx in df.index]
        
    # 1. ローソク足
    fig.add_trace(go.Candlestick(
        x=x_strings,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="価格",
        increasing_line_color="#ef5350",
        decreasing_line_color="#42a5f5"
    ))
    
    fill_colors = {
        3: "rgba(156, 39, 176, 0.05)",
        2: "rgba(33, 150, 243, 0.08)",
        1: "rgba(76, 175, 80, 0.12)"
    }
    line_colors = {
        3: "rgba(156, 39, 176, 0.4)",
        2: "rgba(33, 150, 243, 0.5)",
        1: "rgba(76, 175, 80, 0.6)"
    }
    
    if 'center_line' in df.columns:
        fig.add_trace(go.Scatter(
            x=x_strings, y=df['center_line'],
            line=dict(color="#ffeb3b", width=1.5),
            name="21SMA (センター)"
        ))
        
        for sigma in [3, 2, 1]:
            col_p = f'plus_{sigma}sigma'
            col_m = f'minus_{sigma}sigma'
            
            if col_p in df.columns and col_m in df.columns:
                fig.add_trace(go.Scatter(
                    x=x_strings, y=df[col_p],
                    line=dict(color=line_colors[sigma], width=0.8, dash='dot'),
                    name=f"+{sigma}σ"
                ))
                fig.add_trace(go.Scatter(
                    x=x_strings, y=df[col_m],
                    line=dict(color=line_colors[sigma], width=0.8, dash='dot'),
                    fill='tonexty',
                    fillcolor=fill_colors[sigma],
                    name=f"-{sigma}σ"
                ))
                
    # 2. 遅行スパン
    if 'chikou_span' in df.columns:
        fig.add_trace(go.Scatter(
            x=x_strings, y=df['chikou_span'],
            line=dict(color="#e91e63", width=1.5),
            name="遅行スパン"
        ))
        
    # 3. 売買シグナルマーカーの重畳描画
    if 'signal' in df.columns:
        idx_b_in = np.where(df['signal'] == 1)[0]
        if len(idx_b_in) > 0:
            fig.add_trace(go.Scatter(
                x=[x_strings[i] for i in idx_b_in],
                y=df.iloc[idx_b_in]['Low'] - 0.15,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='#00e676', line=dict(width=1, color='black')),
                name="買いエントリー"
            ))
            
        idx_b_out = np.where(df['signal'] == 2)[0]
        if len(idx_b_out) > 0:
            fig.add_trace(go.Scatter(
                x=[x_strings[i] for i in idx_b_out],
                y=df.iloc[idx_b_out]['High'] + 0.15,
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='#ff9100', line=dict(width=1, color='black')),
                name="買い決済"
            ))
            
        idx_s_in = np.where(df['signal'] == -1)[0]
        if len(idx_s_in) > 0:
            fig.add_trace(go.Scatter(
                x=[x_strings[i] for i in idx_s_in],
                y=df.iloc[idx_s_in]['High'] + 0.15,
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='#ea80fc', line=dict(width=1, color='black')),
                name="売りエントリー"
            ))
            
        idx_s_out = np.where(df['signal'] == -2)[0]
        if len(idx_s_out) > 0:
            fig.add_trace(go.Scatter(
                x=[x_strings[i] for i in idx_s_out],
                y=df.iloc[idx_s_out]['Low'] - 0.15,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='#18ffff', line=dict(width=1, color='black')),
                name="売り決済"
            ))
            
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        dragmode='pan',
        xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=x_strings,
            nticks=8
        )
    )
    
    if len(df) > 0:
        display_count = min(len(df), initial_display_candles)
        sub_df = df.iloc[-display_count:]
        
        start_idx = len(df) - display_count
        end_idx = len(df) - 1
        fig.update_xaxes(range=[start_idx - 0.5, end_idx + 0.5])
        
        y_max_list = [sub_df['High'].max()]
        y_min_list = [sub_df['Low'].min()]
        
        if 'plus_3sigma' in sub_df.columns:
            y_max_list.append(sub_df['plus_3sigma'].max())
        if 'minus_3sigma' in sub_df.columns:
            y_min_list.append(sub_df['minus_3sigma'].min())
            
        if 'chikou_span' in sub_df.columns:
            chk_max = sub_df['chikou_span'].max()
            chk_min = sub_df['chikou_span'].min()
            if not pd.isna(chk_max):
                y_max_list.append(chk_max)
            if not pd.isna(chk_min):
                y_min_list.append(chk_min)
                
        y_max = max(y_max_list)
        y_min = min(y_min_list)
        margin = (y_max - y_min) * 0.05
        
        fig.update_yaxes(range=[y_min - margin, y_max + margin])
        
    return fig

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_fetcher import fetch_data
    from indicator import calculate_super_bollinger
    from backtest import run_backtest
    
    data = fetch_data(period_1h="730d")
    df_1h = data["1h"]
    if not df_1h.empty:
        df_ind = calculate_super_bollinger(df_1h)
        res = run_backtest(df_ind)
        fig = create_super_bollinger_chart(res['df_result'], title="1時間足 スーパーボリンジャー")
        print(f"SUCCESS: Clean chart created with {len(fig.data)} traces.")
