#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成按中心统计的数据集表格（用于SCI论文）
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def identify_center_from_oct_id(oct_id: str) -> Tuple[str, str]:
    """
    从OCT ID识别中心名称和代码
    
    Args:
        oct_id: OCT ID，例如 "M22105_2023_P0000023"
    
    Returns:
        (center_name, center_code) 元组
    """
    if pd.isna(oct_id):
        return 'Unknown', 'Unknown'
    
    oct_str = str(oct_id)
    
    # 中心代码到中心名称的映射（基于代码库中的多个映射规则综合）
    if 'M22105' in oct_str:
        return 'Enshi（恩施）', 'M22105'
    elif 'M22102' in oct_str:
        return 'Xiangyang（襄阳）', 'M22102'
    elif 'M22104' in oct_str:
        return 'Shiyan（十堰）', 'M22104'
    elif 'M22101' in oct_str:
        return 'Jingzhou（荆州）', 'M22101'  # 注意：有些映射中22101是十堰，但这里根据最新规则
    elif 'M0008' in oct_str:
        return 'Jingzhou（荆州）', 'M0008'
    elif 'M20203' in oct_str:
        return 'Wuda（武大）', 'M20203'
    elif 'M20105' in oct_str:
        return 'Wuda（武大）', 'M20105'
    else:
        # 尝试提取数字部分
        import re
        match = re.search(r'M(\d+)', oct_str)
        if match:
            code = match.group(1)
            return f'Unknown({code})', f'M{code}'
        return 'Unknown', 'Unknown'

def analyze_center_distribution(csv_path: Path, subset_name: str) -> pd.DataFrame:
    """
    分析单个数据子集的中心分布
    
    Args:
        csv_path: CSV文件路径
        subset_name: 子集名称（如 "Train", "Val", "External Test"）
    
    Returns:
        包含中心统计信息的DataFrame
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except:
        df = pd.read_csv(csv_path, encoding='gbk')
    
    # 识别中心
    center_info = df['OCT'].apply(identify_center_from_oct_id)
    df['center_name'] = [info[0] for info in center_info]
    df['center_code'] = [info[1] for info in center_info]
    
    # 按中心统计
    center_stats = []
    for center_name in sorted(df['center_name'].unique()):
        center_df = df[df['center_name'] == center_name]
        n_total = len(center_df)
        n_positive = int(center_df['label'].sum())
        n_negative = n_total - n_positive
        positive_rate = (n_positive / n_total * 100) if n_total > 0 else 0.0
        
        center_code = center_df['center_code'].iloc[0] if len(center_df) > 0 else 'Unknown'
        
        center_stats.append({
            'Subset': subset_name,
            'Center': center_name,
            'Center Code': center_code,
            'Total (n)': n_total,
            'Positive (n₊)': n_positive,
            'Negative (n₋)': n_negative,
            'Positive Rate (%)': f'{positive_rate:.1f}%'
        })
    
    return pd.DataFrame(center_stats)

def generate_comprehensive_statistics():
    """生成完整的数据集统计表格"""
    data_root = Path('/data2/hmy/VLM_Caus_Rm_Mics/data/5centers_multi_leave_centers_out')
    
    # 分析各个子集
    subsets = {
        'Train (internal)': data_root / 'train_labels.csv',
        'Val (internal)': data_root / 'val_labels.csv',
        'External Test': data_root / 'external_test_labels.csv'
    }
    
    all_stats = []
    for subset_name, csv_path in subsets.items():
        if csv_path.exists():
            stats_df = analyze_center_distribution(csv_path, subset_name)
            all_stats.append(stats_df)
        else:
            print(f"⚠️ 文件不存在: {csv_path}")
    
    # 合并所有统计
    combined_df = pd.concat(all_stats, ignore_index=True)
    
    # 生成汇总表（按中心汇总所有子集）
    center_summary = []
    for center_name in sorted(combined_df['Center'].unique()):
        center_data = combined_df[combined_df['Center'] == center_name]
        total_n = center_data['Total (n)'].sum()
        total_pos = center_data['Positive (n₊)'].sum()
        total_neg = total_n - total_pos
        overall_rate = (total_pos / total_n * 100) if total_n > 0 else 0.0
        
        center_code = center_data['Center Code'].iloc[0] if len(center_data) > 0 else 'Unknown'
        
        # 统计各子集的分布
        train_n = center_data[center_data['Subset'] == 'Train (internal)']['Total (n)'].sum() if 'Train (internal)' in center_data['Subset'].values else 0
        val_n = center_data[center_data['Subset'] == 'Val (internal)']['Total (n)'].sum() if 'Val (internal)' in center_data['Subset'].values else 0
        test_n = center_data[center_data['Subset'] == 'External Test']['Total (n)'].sum() if 'External Test' in center_data['Subset'].values else 0
        
        center_summary.append({
            'Center': center_name,
            'Center Code': center_code,
            'Train (n)': int(train_n),
            'Val (n)': int(val_n),
            'External Test (n)': int(test_n),
            'Total (n)': int(total_n),
            'Positive (n₊)': int(total_pos),
            'Negative (n₋)': int(total_neg),
            'Positive Rate (%)': f'{overall_rate:.1f}%'
        })
    
    summary_df = pd.DataFrame(center_summary)
    
    return combined_df, summary_df

def generate_latex_tables(detailed_df: pd.DataFrame, summary_df: pd.DataFrame):
    """生成LaTeX格式的表格"""
    
    # 表1：详细分布表（按子集和中心）
    latex_detailed = """
\\begin{table}[t]
\\centering
\\caption{Detailed center-wise distribution across data subsets.}
\\label{tab:center_distribution_detailed}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lllcccc}
\\toprule
Subset & Center & Code & Total $(n)$ & Positive $(n_{+})$ & Negative $(n_{-})$ & Positive Rate (\\%) \\\\
\\midrule
"""
    
    for _, row in detailed_df.iterrows():
        latex_detailed += f"{row['Subset']} & {row['Center']} & {row['Center Code']} & {row['Total (n)']} & {row['Positive (n₊)']} & {row['Negative (n₋)']} & {row['Positive Rate (%)']} \\\\\n"
    
    latex_detailed += """\\bottomrule
\\end{tabular}%
}
\\end{table}
"""
    
    # 表2：汇总表（按中心汇总）
    latex_summary = """
\\begin{table}[t]
\\centering
\\caption{Center-wise summary statistics of the five-center multimodal cervical lesion dataset.}
\\label{tab:center_summary}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llccccccl}
\\toprule
Center & Code & Train $(n)$ & Val $(n)$ & External Test $(n)$ & Total $(n)$ & Positive $(n_{+})$ & Negative $(n_{-})$ & Positive Rate (\\%) \\\\
\\midrule
"""
    
    for _, row in summary_df.iterrows():
        latex_summary += f"{row['Center']} & {row['Center Code']} & {row['Train (n)']} & {row['Val (n)']} & {row['External Test (n)']} & {row['Total (n)']} & {row['Positive (n₊)']} & {row['Negative (n₋)']} & {row['Positive Rate (%)']} \\\\\n"
    
    latex_summary += """\\bottomrule
\\end{tabular}%
}
\\end{table}
"""
    
    return latex_detailed, latex_summary

def main():
    """主函数"""
    print("=" * 80)
    print("生成按中心统计的数据集表格（用于SCI论文）")
    print("=" * 80)
    
    # 生成统计
    detailed_df, summary_df = generate_comprehensive_statistics()
    
    # 保存CSV
    output_dir = Path(__file__).parent
    detailed_df.to_csv(output_dir / 'center_distribution_detailed.csv', index=False, encoding='utf-8-sig')
    summary_df.to_csv(output_dir / 'center_summary.csv', index=False, encoding='utf-8-sig')
    
    print("\n📊 详细分布表（按子集和中心）：")
    print(detailed_df.to_string(index=False))
    
    print("\n📊 汇总表（按中心汇总）：")
    print(summary_df.to_string(index=False))
    
    # 生成LaTeX表格
    latex_detailed, latex_summary = generate_latex_tables(detailed_df, summary_df)
    
    # 保存LaTeX文件
    with open(output_dir / 'center_statistics_latex.tex', 'w', encoding='utf-8') as f:
        f.write("% 表1：详细分布表\n")
        f.write(latex_detailed)
        f.write("\n\n% 表2：汇总表\n")
        f.write(latex_summary)
    
    print("\n✅ 统计表格已生成：")
    print(f"   - center_distribution_detailed.csv")
    print(f"   - center_summary.csv")
    print(f"   - center_statistics_latex.tex")
    
    # 打印LaTeX表格
    print("\n" + "=" * 80)
    print("LaTeX表格（可直接复制到论文中）：")
    print("=" * 80)
    print("\n【表1：详细分布表】")
    print(latex_detailed)
    print("\n【表2：汇总表】")
    print(latex_summary)

if __name__ == "__main__":
    main()

