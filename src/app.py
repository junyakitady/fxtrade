import streamlit as st
import pandas as pd
import yfinance as yf
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
    data = fetch_data(period_1h="730d", period_1d="5y")
    results = {}
    for tf_code, df in data.items():
        if not df.empty:
            df_ind = calculate_super_bollinger(df)
            res = run_backtest(df_ind, timeframe=tf_code, volume=10000)
            results[tf_code] = res
        else:
            results[tf_code] = None
            
    try:
        df_goog = yf.Ticker("GOOG").history(period="5y", interval="1d")
        if not df_goog.empty:
            df_clean = df_goog[['Open', 'High', 'Low', 'Close']].dropna()
            df_ind_goog = calculate_super_bollinger(df_clean)
            res_goog = run_backtest(df_ind_goog, timeframe="1h", volume=100)
            results["GOOG"] = res_goog
        else:
            results["GOOG"] = None
    except Exception:
        results["GOOG"] = None
        
    return results

def render_signal_badge(tf_name: str, res: dict, strat_desc: str = ""):
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
        
    st.markdown(f"**{tf_name}**: &nbsp; {badge_html}", unsafe_allow_html=True)
    if strat_desc:
        # ポジションアイコンの直下に、インデントされた少し控えめな文字色で戦略を美しく配置
        st.markdown(
            f"<div style='color:#b0bec5; font-size:0.85em; margin-left:1rem; margin-bottom:0.6rem;'>"
            f"↳ 戦略: {strat_desc}"
            f"</div>",
            unsafe_allow_html=True
        )

def main():
    st.html(
        """
        <script>
            setTimeout(function(){
                window.parent.location.reload();
            }, 3600000);
        </script>
        """
    )
    
    st.title("📈 スーパーボリンジャー トレード支援エージェント")
    st.markdown(
        "各時間軸の相場特性に合わせて厳選された特化型ロジック（マルチ・ロジック戦略）による"
        "独立した売買シグナル判定と成績表を自動更新で統合表示します。"
    )
    
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    
    with st.spinner("市場データを取得・解析中..."):
        results = load_and_process_data()
        
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("📊 為替 総合サマリー (USD/JPY)")
        with st.container(border=True):
            st.markdown("#### 各時間軸のステータスと適用戦略")
            
            strategy_descriptions = {
                "1h": "入:遅行&+2σ&拡大 ｜ 出:遅行逆転",
                "4h": "入:遅行&+1σ ｜ 出:1σ割れ",
                "1d": "入:遅行&+1σ&拡大 ｜ 出:中心線割れ"
            }
            
            render_signal_badge("1時間足", results.get("1h"), strategy_descriptions["1h"])
            render_signal_badge("4時間足", results.get("4h"), strategy_descriptions["4h"])
            render_signal_badge("日足", results.get("1d"), strategy_descriptions["1d"])
            
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
                fig_1h = create_super_bollinger_chart(res_1h['df_result'], "1時間足 スーパーボリンジャー")
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
                fig_4h = create_super_bollinger_chart(res_4h['df_result'], "4時間足 スーパーボリンジャー")
                st.plotly_chart(fig_4h, width='stretch')
            else:
                st.info("データがありません。")
                
    with row2_col2:
        st.subheader("📅 日足")
        with st.container(border=True):
            res_1d = results.get("1d")
            if res_1d:
                fig_1d = create_super_bollinger_chart(res_1d['df_result'], "日足 スーパーボリンジャー")
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
                render_signal_badge("GOOG", res_goog)
                st.markdown("<br>", unsafe_allow_html=True)
                
                pos_g = res_goog['current_position']
                sig_g = "🟢 買い保有中" if pos_g == 1 else "🔴 売り保有中" if pos_g == -1 else "⚪ ノーポジション"
                l_g = res_goog.get('long_trades', 0)
                s_g = res_goog.get('short_trades', 0)
                
                st.markdown(f"**シグナル判定**: {sig_g}")
                st.markdown("**適用戦略**: 入:遅行&+2σ&拡大 ｜ 出:遅行逆転 (モメンタム特化)")
                st.markdown(f"**総取引回数**: {res_goog['total_trades']} 回 (買:{l_g} / 売:{s_g})")
                st.markdown(f"**勝率**: {res_goog['win_rate']:.1f} %")
                st.markdown(f"**合計損益**: ${res_goog['total_profit']:,.2f} (単利100株固定)")
                if res_goog['total_trades'] > 0:
                    st.markdown(f"**平均損益**: ${res_goog['total_profit']/res_goog['total_trades']:,.2f}")
                    
            with col_g_chart:
                fig_goog = create_super_bollinger_chart(res_goog['df_result'], "GOOG 日足 スーパーボリンジャー")
                st.plotly_chart(fig_goog, width='stretch')
        else:
            st.info("GOOGデータがありません。")

if __name__ == "__main__":
    main()
