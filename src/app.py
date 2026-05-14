import streamlit as st
import streamlit.components.v1 as components
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
    data = fetch_data(period_1h="730d", period_1d="5y")
    results = {}
    for tf, df in data.items():
        if not df.empty:
            df_ind = calculate_super_bollinger(df)
            res = run_backtest(df_ind, volume=10000)
            results[tf] = res
        else:
            results[tf] = None
    return results

def render_signal_badge(tf_name: str, res: dict):
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

def main():
    components.html(
        """
        <script>
            setTimeout(function(){
                window.parent.location.reload();
            }, 3600000);
        </script>
        """,
        height=0, width=0
    )
    
    st.title("📈 スーパーボリンジャー トレード支援エージェント")
    st.markdown(
        "各時間軸のスーパーボリンジャーチャート（初期表示：最新64本）と、"
        "独立した売買シグナル判定および成績を自動更新で統合表示します。"
    )
    
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    
    with st.spinner("市場データを取得・解析中..."):
        results = load_and_process_data()
        
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("📊 総合サマリー")
        with st.container(border=True):
            st.markdown("#### 各時間軸のステータスと成績")
            
            render_signal_badge("1時間足", results.get("1h"))
            st.markdown("<br>", unsafe_allow_html=True)
            render_signal_badge("4時間足", results.get("4h"))
            st.markdown("<br>", unsafe_allow_html=True)
            render_signal_badge("日足", results.get("1d"))
            
            st.markdown("---")
            
            summary_rows = []
            for tf_code, tf_label in [("1h", "1時間足"), ("4h", "4時間足"), ("1d", "日足")]:
                r = results.get(tf_code)
                if r:
                    pos = r['current_position']
                    if pos == 1:
                        sig_str = "🟢 買い保有中"
                    elif pos == -1:
                        sig_str = "🔴 売り保有中"
                    else:
                        sig_str = "⚪ ノーポジション"
                        
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
            st.dataframe(df_summary, hide_index=True, use_container_width=True)
            
    with row1_col2:
        st.subheader("🕒 1時間足")
        with st.container(border=True):
            res_1h = results.get("1h")
            if res_1h:
                fig_1h = create_super_bollinger_chart(res_1h['df_result'], "1時間足 スーパーボリンジャー")
                st.plotly_chart(fig_1h, use_container_width=True)
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
                st.plotly_chart(fig_4h, use_container_width=True)
            else:
                st.info("データがありません。")
                
    with row2_col2:
        st.subheader("📅 日足")
        with st.container(border=True):
            res_1d = results.get("1d")
            if res_1d:
                fig_1d = create_super_bollinger_chart(res_1d['df_result'], "日足 スーパーボリンジャー")
                st.plotly_chart(fig_1d, use_container_width=True)
            else:
                st.info("データがありません。")

if __name__ == "__main__":
    main()
