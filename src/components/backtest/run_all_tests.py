import os
import sys
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 親のsrcディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_loader import get_usd_jpy_data
from indicator import calculate_super_bollinger, calculate_macd
from backtest import run_backtest

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')

# 戦略の説明マッピング
ENTRY_DESC = {
    'E1': '遅行スパン陽転のみ（エクスパンションなし）',
    'E2': 'E1 & 終値 > +1σ',
    'E3': 'E1 & 終値 > +2σ',
    'E4': '遅行スパン陽転 & エクスパンション（バンド拡大）',
    'E5': 'E4 & 終値 > +1σ',
    'E6': 'E4 & 終値 > +2σ',
    'E1_MACD': 'E1 & MACDゴールデンクロス',
    'E2_MACD': 'E2 & MACDゴールデンクロス',
    'E3_MACD': 'E3 & MACDゴールデンクロス',
    'E4_MACD': 'E4 & MACDゴールデンクロス',
    'E5_MACD': 'E5 & MACDゴールデンクロス',
    'E6_MACD': 'E6 & MACDゴールデンクロス',
    'MACD_DOTEN': 'MACDのみのドテン'
}

EXIT_DESC = {
    'EX1': '遅行スパン陰転',
    'EX2': '終値 < +1σ割れ',
    'EX3': '終値 < 21MA割れ',
    'EX4': '遅行スパン陰転 or 終値 < +1σ割れ',
    'EX5': '遅行スパン陰転 or 21MA割れ',
    'EX6': 'MACD逆クロス (ロング: デッドクロス / ショート: ゴールデンクロス)',
    'EX1_MACD': 'EX1 or MACD逆クロス (ロング: デッドクロス / ショート: ゴールデンクロス)',
    'EX2_MACD': 'EX2 or MACD逆クロス (ロング: デッドクロス / ショート: ゴールデンクロス)',
    'EX3_MACD': 'EX3 or MACD逆クロス (ロング: デッドクロス / ショート: ゴールデンクロス)',
    'EX4_MACD': 'EX4 or MACD逆クロス (ロング: デッドクロス / ショート: ゴールデンクロス)',
    'EX5_MACD': 'EX5 or MACD逆クロス (ロング: デッドクロス / ショート: ゴールデンクロス)',
    'EX7': '遅行スパン陰転 or 21MA割れ or ±1σ割れ',
    'EX7_MACD': 'EX7 or MACD逆クロス (ロング: デッドクロス / ショート: ゴールデンクロス)',
    'MACD_DOTEN': 'MACDのみのドテン'
}

def run_all_combinations():
    intervals = ['1h', '4h', '1d']
    periods = {
        '1h': '2y',
        '4h': '2y',
        '1d': '5y'
    }
    
    # エントリー戦略 12パターン
    entry_strategies = [
        'E1', 'E2', 'E3', 'E4', 'E5', 'E6',
        'E1_MACD', 'E2_MACD', 'E3_MACD', 'E4_MACD', 'E5_MACD', 'E6_MACD'
    ]
    
    # エグジット戦略 13パターン (EX7, EX7_MACDを追加)
    exit_strategies = [
        'EX1', 'EX2', 'EX3', 'EX4', 'EX5', 'EX6',
        'EX1_MACD', 'EX2_MACD', 'EX3_MACD', 'EX4_MACD', 'EX5_MACD',
        'EX7', 'EX7_MACD'
    ]
    
    all_results = []
    best_results = {} # 各時間軸の最良結果を保存
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    for interval in intervals:
        period = periods[interval]
        print(f"\n=========================================")
        print(f"Processing Interval: {interval} (Period: {period})")
        print(f"=========================================")
        
        # データ取得と指標計算
        df = get_usd_jpy_data(interval, period)
        df_ind = calculate_super_bollinger(df)
        df_ind = calculate_macd(df_ind) # MACDも算出
        
        interval_results = []
        
        # 1. ボリンジャーバンド戦略 (12 x 10 ＝ 120パターン、遅行スパンはhigh_low固定)
        for entry in entry_strategies:
            for exit_strat in exit_strategies:
                # バックテスト実行
                result = run_backtest(df_ind, entry, exit_strat, lagging_type='high_low')
                metrics = result['metrics']
                
                if metrics is None or metrics['total_trades'] == 0:
                    continue
                    
                res_dict = {
                    'interval': interval,
                    'lagging_type': 'high_low',
                    'lagging_desc': '高安値基準 (Close vs High/Low_21_ago)',
                    'entry': entry,
                    'exit': exit_strat,
                    'entry_desc': ENTRY_DESC[entry],
                    'exit_desc': EXIT_DESC[exit_strat],
                    'total_trades': metrics['total_trades'],
                    'win_rate': metrics['win_rate'],
                    'total_pnl': metrics['total_pnl'],
                    'avg_pnl': metrics['avg_pnl'],
                    'profit_factor': metrics['profit_factor'],
                    'max_drawdown': metrics['max_drawdown'],
                    'win_trades': metrics['win_trades'],
                    'lose_trades': metrics['lose_trades']
                }
                
                # 評価スコア
                return_dd_ratio = res_dict['total_pnl'] / res_dict['max_drawdown'] if res_dict['max_drawdown'] > 0 else res_dict['total_pnl']
                res_dict['return_dd_ratio'] = return_dd_ratio
                
                pf = metrics['profit_factor']
                pf_val = 99.9 if (pf == float('inf') or np.isinf(pf)) else pf
                res_dict['pf_over_trades'] = pf_val / metrics['total_trades'] if metrics['total_trades'] > 0 else 0.0
                
                res_dict['_balance_history'] = result['balance_history']
                res_dict['_trades'] = result['trades']
                
                interval_results.append(res_dict)
                all_results.append(res_dict)
                
        # 2. MACDドテン戦略 (1パターン)
        result_macd = run_backtest(df_ind, entry_strat="MACD_DOTEN", exit_strat="MACD_DOTEN", lagging_type="none")
        metrics_macd = result_macd['metrics']
        
        if metrics_macd is not None and metrics_macd['total_trades'] > 0:
            res_dict_macd = {
                'interval': interval,
                'lagging_type': 'none',
                'lagging_desc': 'MACDドテン（遅行スパン不要）',
                'entry': 'MACD_DOTEN',
                'exit': 'MACD_DOTEN',
                'entry_desc': ENTRY_DESC['MACD_DOTEN'],
                'exit_desc': EXIT_DESC['MACD_DOTEN'],
                'total_trades': metrics_macd['total_trades'],
                'win_rate': metrics_macd['win_rate'],
                'total_pnl': metrics_macd['total_pnl'],
                'avg_pnl': metrics_macd['avg_pnl'],
                'profit_factor': metrics_macd['profit_factor'],
                'max_drawdown': metrics_macd['max_drawdown'],
                'win_trades': metrics_macd['win_trades'],
                'lose_trades': metrics_macd['lose_trades']
            }
            
            # 評価スコア
            return_dd_ratio = res_dict_macd['total_pnl'] / res_dict_macd['max_drawdown'] if res_dict_macd['max_drawdown'] > 0 else res_dict_macd['total_pnl']
            res_dict_macd['return_dd_ratio'] = return_dd_ratio
            
            pf = metrics_macd['profit_factor']
            pf_val = 99.9 if (pf == float('inf') or np.isinf(pf)) else pf
            res_dict_macd['pf_over_trades'] = pf_val / metrics_macd['total_trades'] if metrics_macd['total_trades'] > 0 else 0.0
            
            res_dict_macd['_balance_history'] = result_macd['balance_history']
            res_dict_macd['_trades'] = result_macd['trades']
            
            interval_results.append(res_dict_macd)
            all_results.append(res_dict_macd)
            
        if not interval_results:
            print(f"No successful tests for interval {interval}")
            continue
            
        # データフレーム化してソート
        df_interval = pd.DataFrame(interval_results)
        
        # 最良戦略の選定基準:
        # 1. トレード数5回以上
        # 2. その中で 合計損益 (total_pnl) が最大のもの
        valid_strats = df_interval[df_interval['total_trades'] >= 5]
        if not valid_strats.empty:
            best_strat = valid_strats.sort_values('total_pnl', ascending=False).iloc[0]
        else:
            best_strat = df_interval.sort_values('total_pnl', ascending=False).iloc[0]
            
        best_results[interval] = best_strat
        
        # 最良戦略の資産曲線をプロット
        plot_balance_chart(interval, best_strat)
        
    # 全結果をCSV保存
    df_all = pd.DataFrame(all_results)
    df_all_save = df_all.drop(columns=['_balance_history', '_trades'], errors='ignore')
    csv_path = os.path.join(REPORTS_DIR, 'all_results.csv')
    df_all_save.to_csv(csv_path, index=False)
    print(f"\nSaved all results to {csv_path}")
    
    # レポート作成
    generate_markdown_report(df_all, best_results)

def plot_balance_chart(interval: str, best_strat: pd.Series):
    """
    最良戦略の資産推移グラフを保存する。
    """
    balance_df = best_strat['_balance_history']
    if balance_df is None or balance_df.empty:
        return
        
    plt.figure(figsize=(10, 5))
    plt.step(balance_df['datetime'], balance_df['balance'], where='post', label='Capital Balance', color='blue', linewidth=2)
    
    plt.title(f"Best Strategy Asset Curve ({interval})\nEntry: {best_strat['entry']}, Exit: {best_strat['exit']}, Lagging: {best_strat['lagging_type']}")
    plt.xlabel('Date')
    plt.ylabel('Capital (JPY)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    chart_path = os.path.join(REPORTS_DIR, f'balance_chart_{interval}.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved best strategy chart for {interval} to {chart_path}")

def generate_markdown_report(df_all: pd.DataFrame, best_results: dict):
    """
    バックテスト結果をまとめたMarkdownレポートを生成する。
    """
    report_path = os.path.join(REPORTS_DIR, 'backtest_report.md')
    
    df_clean = df_all.drop(columns=['_balance_history', '_trades'], errors='ignore')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# スーパーボリンジャー取引戦略 一括バックテスト検証レポート (High-Low基準限定)\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 概要
        f.write("## 概要\n")
        f.write("ドル円 (USD/JPY) の為替取引において、ボリンジャーバンドとMACDを融合させた取引戦略の一括バックテスト検証を行いました。\n")
        f.write("エントリー条件 12種 × エグジット条件 10種（＝120パターン）に加えて、MACDドテン戦略1パターンを足した計121パターンを、3つの時間軸（1時間足、4時間足、1日足）で比較検証しました（合計363検証パターン）。\n\n")
        f.write("遅行スパン基準は、ダマシを少なくし堅牢なトレンドフォローを狙う目的から、すべて**高安値基準 (high_low)**に固定しています。\n\n")
        
        f.write("### 取引基本条件\n")
        f.write("- **対象ペア**: USD/JPY (`JPY=X` via yFinance)\n")
        f.write("- **取引数量**: 1万通貨固定 (同時に最大1ポジションのみ、ドテンなしの個別決済。ただしMACDドテンのみドテンあり)\n")
        f.write("- **初期資金**: 1,000,000円\n")
        f.write("- **検証期間**:\n")
        f.write("  - 1時間足 (1h): 過去2年間\n")
        f.write("  - 4時間足 (4h): 過去2年間 (1hからリサンプル)\n")
        f.write("  - 1日足 (1d): 過去5年間 (始終値スマート補正あり)\n\n")
        
        # 最良のまとめ
        f.write("## 各時間軸における最良戦略のまとめ（合計損益 基準）\n")
        f.write("※信頼性確保のため、総トレード数が5回以上の戦略から選定しています。\n\n")
        
        for interval in ['1h', '4h', '1d']:
            if interval not in best_results:
                continue
            best = best_results[interval]
            f.write(f"### ■ {interval.upper()} 足における最良戦略: {best['entry']} & {best['exit']}\n")
            f.write(f"- **遅行スパン判定基準**: {best['lagging_desc']}\n")
            f.write(f"- **エントリー戦略**: {best['entry']} ({best['entry_desc']})\n")
            f.write(f"- **エグジット戦略**: {best['exit']} ({best['exit_desc']})\n")
            f.write(f"- **総トレード数**: {best['total_trades']} 回\n")
            f.write(f"- **勝率**: {best['win_rate']:.2%}\n")
            f.write(f"- **プロフィットファクター (PF)**: {best['profit_factor']:.2f}\n")
            f.write(f"- **PF / 取引回数**: **{best['pf_over_trades']:.5f}**\n")
            f.write(f"- **合計損益**: **{best['total_pnl']:,.0f} 円**\n")
            f.write(f"- **平均損益/トレード**: {best['avg_pnl']:,.0f} 円\n")
            f.write(f"- **最大ドローダウン (DD)**: {best['max_drawdown']:,.0f} 円\n")
            f.write(f"- **損益/DD比率 (Return/DD)**: {best['return_dd_ratio']:.2f}\n\n")
            
            chart_filename = f"balance_chart_{interval}.png"
            f.write(f"![{interval.upper()} Best Strategy Asset Curve]({chart_filename})\n\n")
            f.write("---\n\n")
            
        # 時間軸別の全戦略比較データ（合計損益 順、上位15位まで）
        f.write("## 時間軸別の全戦略比較データ（合計損益 順、上位15位まで）\n")
        f.write("各時間軸において、合計損益が高い上位15戦略を表示します。(詳細データは CSV にて保存されています)\n\n")
        
        for interval in ['1h', '4h', '1d']:
            f.write(f"### {interval.upper()} 足 上位15戦略一覧\n")
            df_int = df_clean[df_clean['interval'] == interval].sort_values('total_pnl', ascending=False).head(15)
            
            # テーブルヘッダー
            f.write("| 順位 | エントリー | エグジット | トレード数 | 勝率 | PF / 取引回数 | PF | 合計損益 (円) | 最大DD (円) |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            
            for idx, (_, row) in enumerate(df_int.iterrows(), 1):
                pf_str = f"{row['profit_factor']:.2f}" if row['profit_factor'] != float('inf') else "inf"
                f.write(f"| {idx} | {row['entry']} | {row['exit']} | {row['total_trades']} | {row['win_rate']:.1%} | {row['pf_over_trades']:.5f} | {pf_str} | {row['total_pnl']:,.0f} | {row['max_drawdown']:,.0f} |\n")
            f.write("\n\n")
            
        # 戦略コード表
        f.write("## 戦略コード表\n")
        f.write("### エントリー戦略\n")
        for k, v in ENTRY_DESC.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n### エグジット戦略\n")
        for k, v in EXIT_DESC.items():
            f.write(f"- **{k}**: {v}\n")
            
    print(f"Generated markdown report at {report_path}")

if __name__ == "__main__":
    run_all_combinations()
