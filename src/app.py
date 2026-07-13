import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import fetch_data
from indicator import calculate_super_bollinger
from backtest import run_backtest
from components.charts import create_super_bollinger_chart

st.set_page_config(
    page_title="スーパーボリンジャー トレードエージェント",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_and_process_data():
    data = fetch_data(period_1h="2y", period_1d="5y")
    results = {}
    
    # 1時間足データの処理
    df_1h = data.get("1h")
    if df_1h is not None and not df_1h.empty:
        df_ind_1h = calculate_super_bollinger(df_1h)
        # E6 + EX2 (厳選・低DDモデル)
        results["1h"] = run_backtest(df_ind_1h, timeframe="1h_e6_ex2", volume=10000)
    else:
        results["1h"] = None
        
    # 4時間足データの処理 (1時間足から自動生成される)
    df_4h = data.get("4h")
    if df_4h is not None and not df_4h.empty:
        df_ind_4h = calculate_super_bollinger(df_4h)
        # E1 + EX3 (エクスパンションなし)
        results["4h"] = run_backtest(df_ind_4h, timeframe="4h_e1_ex3", volume=10000)
    else:
        results["4h"] = None
        
    # 日足データの処理
    df_1d = data.get("1d")
    if df_1d is not None and not df_1d.empty:
        df_ind_1d = calculate_super_bollinger(df_1d)
        # E2 + EX5 (エクスパンションなし)
        results["1d"] = run_backtest(df_ind_1d, timeframe="1d_e2_ex5", volume=10000)
    else:
        results["1d"] = None
        
    try:
        goog_data = fetch_data(symbol="GOOG", period_1d="5y", raise_errors=True)
        df_goog = goog_data.get("1d")
        if df_goog is not None and not df_goog.empty:
            df_clean = df_goog[['Open', 'High', 'Low', 'Close']].dropna()
            df_ind_goog = calculate_super_bollinger(df_clean)
            res_goog = run_backtest(df_ind_goog, timeframe="goog_e4_ex5", volume=100)
            results["GOOG"] = res_goog
        else:
            results["GOOG"] = None
    except Exception:
        results["GOOG"] = None
        
    return results

def render_signal_badge(tf_name: str, res: dict, entry_desc: str = "", exit_desc: str = "", is_usd: bool = False):
    if res is None:
        st.markdown(f"**{tf_name}**: データなし")
        return
        
    pos_code = res['current_position']
    if pos_code == 1:
        badge_html = (
            '<span style="background-color:#00e676; color:black; padding:4px 12px; '
            'border-radius:16px; font-weight:bold; font-size:0.9em;">'
            '🟢 買いポジション保有中'
            '</span>'
        )
    elif pos_code == -1:
        badge_html = (
            '<span style="background-color:#ff5252; color:white; padding:4px 12px; '
            'border-radius:16px; font-weight:bold; font-size:0.9em;">'
            '🔴 売りポジション保有中'
            '</span>'
        )
    else:
        badge_html = (
            '<span style="background-color:#424242; color:white; padding:4px 12px; '
            'border-radius:16px; font-weight:bold; font-size:0.9em;">'
            '⚪ ノーポジション'
            '</span>'
        )
        
    # 保有ポジションがある場合、エントリー建値とリアルタイム評価損益を表示
    detail_html = ""
    if pos_code != 0:
        entry_price = res.get('current_entry_price', 0.0)
        unrealized_profit = res.get('current_unrealized_profit', 0.0)
        
        if is_usd:
            entry_str = f"\\${entry_price:,.2f}"
            profit_str = f"+\\${unrealized_profit:,.2f}" if unrealized_profit >= 0 else f"-\\${abs(unrealized_profit):,.2f}"
        else:
            entry_str = f"{entry_price:,.3f}円"
            profit_str = f"{unrealized_profit:+,.0f}円"
            
        profit_color = "#00e676" if unrealized_profit >= 0 else "#ff8a80"
        
        detail_html = (
            f'&nbsp;&nbsp;&nbsp;&nbsp;'
            f'<span style="font-size:0.9em; color:#cfd8dc;">'
            f'建値: <b style="color:#ffffff;">{entry_str}</b>'
            f' &nbsp;➔&nbsp; '
            f'評価損益: <b style="color:{profit_color}; font-size:1.05em;">{profit_str}</b>'
            f'</span>'
        )
        
    st.markdown(f"**{tf_name}**: &nbsp; {badge_html}{detail_html}", unsafe_allow_html=True)
    if entry_desc:
        st.markdown(
            f"<div style='color:#b0bec5; font-size:0.85em; margin-left:1rem; margin-bottom:0.1rem;'>"
            f"↳ エントリー: {entry_desc}"
            f"</div>",
            unsafe_allow_html=True
        )
    if exit_desc:
        st.markdown(
            f"<div style='color:#b0bec5; font-size:0.85em; margin-left:1rem; margin-bottom:0.6rem;'>"
            f"↳ エグジッド: {exit_desc}"
            f"</div>",
            unsafe_allow_html=True
        )

@st.fragment(run_every=3600)  # 1時間 (3600秒) ごとに自動再評価・再描画
def render_dashboard():
    with st.spinner("市場データを取得・解析中..."):
        results = load_and_process_data()
        
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("📊 為替 総合サマリー (USD/JPY)")
        with st.container(border=True):
            st.markdown("#### 各時間軸のステータスと適用戦略")
            
            strategy_descriptions = {
                "1h": {"entry_desc": "遅行スパン陽転 & バンド拡大 & ±2σ 越え", "exit_desc": "±1σ割れ"},
                "4h": {"entry_desc": "遅行スパン陽転のみ", "exit_desc": "21MA割れ"},
                "1d": {"entry_desc": "遅行スパン陽転 & ±1σ 越え", "exit_desc": "遅行スパン陰転 or 21MA割れ"}
            }
            
            render_signal_badge("1時間足", results.get("1h"), **strategy_descriptions["1h"])
            render_signal_badge("4時間足", results.get("4h"), **strategy_descriptions["4h"])
            render_signal_badge("日足", results.get("1d"), **strategy_descriptions["1d"])
            
            st.markdown("---")
            
            summary_rows = []
            for tf_code, tf_label in [("1h", "1時間足"), ("4h", "4時間足"), ("1d", "日足")]:
                r = results.get(tf_code)
                if r:
                    pos = r['current_position']
                    sig_str = "🟢 買い保有中" if pos == 1 else "🔴 売り保有中" if pos == -1 else "⚪ ノーポジション"
                    l_cnt = r.get('long_trades', 0)
                    s_cnt = r.get('short_trades', 0)
                    
                    summary_rows.append({
                        "時間軸": tf_label,
                        "シグナル判定": sig_str,
                        "総取引回数 (買/売)": f"{r['total_trades']} 回 (買:{l_cnt} / 売:{s_cnt})",
                        "勝率": f"{r['win_rate']:.1f} %",
                        "損益合計": f"{r['total_profit']:,.0f} 円"
                    })
                else:
                    summary_rows.append({
                        "時間軸": tf_label,
                        "シグナル判定": "データなし",
                        "総取引回数 (買/売)": "-", "勝率": "-", "損益合計": "-"
                    })
                    
            df_summary = pd.DataFrame(summary_rows)
            # 表側からは戦略列を除去し、横幅をスッキリ確保
            st.dataframe(df_summary, hide_index=True, width='stretch')
            
    with row1_col2:
        st.subheader("🕒 1時間足")
        with st.container(border=True):
            res_1h = results.get("1h")
            if res_1h:
                fig_1h = create_super_bollinger_chart(res_1h['df_result'], "")
                st.plotly_chart(fig_1h, width='stretch')
            else:
                st.info("データがありません。")
                
    st.markdown("<br>", unsafe_allow_html=True)
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.subheader("🕓 4時間足")
        with st.container(border=True):
            res_4h = results.get("4h")
            if res_4h:
                fig_4h = create_super_bollinger_chart(res_4h['df_result'], "")
                st.plotly_chart(fig_4h, width='stretch')
            else:
                st.info("データがありません。")
                
    with row2_col2:
        st.subheader("📅 日足")
        with st.container(border=True):
            res_1d = results.get("1d")
            if res_1d:
                fig_1d = create_super_bollinger_chart(res_1d['df_result'], "")
                st.plotly_chart(fig_1d, width='stretch')
            else:
                st.info("データがありません。")

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("🇺🇸 米国個別株 トレンド分析 (GOOG)")
    
    with st.container(border=True):
        res_goog = results.get("GOOG")
        if res_goog:
            col_g_sum, col_g_chart = st.columns([1, 2])
            
            with col_g_sum:
                st.markdown("#### GOOG 日足ステータス")
                
                strategy_descriptions_goog = {
                    "entry_desc": "遅行スパン陽転（終値比較） & バンド拡大",
                    "exit_desc": "遅行スパン陰転（終値比較） or 21MA割れ"
                }
                
                render_signal_badge("GOOG", res_goog, **strategy_descriptions_goog, is_usd=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                l_g = res_goog.get('long_trades', 0)
                s_g = res_goog.get('short_trades', 0)
                
                summary_goog_rows = [{
                    "銘柄": "GOOG",
                    "総取引回数 (買/売)": f"{res_goog['total_trades']} 回 (買:{l_g} / 売:{s_g})",
                    "勝率": f"{res_goog['win_rate']:.1f} %",
                    "損益合計": f"${res_goog['total_profit']:,.2f}"
                }]
                df_g_summary = pd.DataFrame(summary_goog_rows)
                st.dataframe(df_g_summary, hide_index=True, width='stretch')
                    
            with col_g_chart:
                fig_goog = create_super_bollinger_chart(res_goog['df_result'], "")
                st.plotly_chart(fig_goog, width='stretch')
        else:
            st.info("GOOGデータがありません。")

def main():
    st.title("📈 スーパーボリンジャー トレード支援エージェント")
    st.markdown(
        "各時間軸の相場特性に合わせて厳選された特化型ロジック（マルチ・ロジック戦略）による"
        "独立した売買シグナル判定と成績表を自動更新で統合表示します。"
    )
    
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    
    # 自動更新フラグメントの実行
    render_dashboard()

if __name__ == "__main__":
    main()
